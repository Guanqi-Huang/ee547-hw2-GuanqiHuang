import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError, NoRegionError

# Helpers
def utc_iso(dt: Optional[datetime] = None) -> str:
    return (dt or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
def warn(msg: str): print(f"[WARNING] {msg}", file=sys.stderr)
def err(msg: str):  print(f"[ERROR] {msg}", file=sys.stderr)
def make_session(region: Optional[str]):
    cfg = Config(retries={"max_attempts": 2, "mode": "standard"}, connect_timeout=4, read_timeout=20)
    if region:
        return boto3.Session(region_name=region), cfg
    return boto3.Session(), cfg

def verify_auth(sess) -> Dict[str, str]:
    sts = sess.client("sts")
    ident = sts.get_caller_identity()
    return {
        "account_id": ident["Account"],
        "user_arn": ident["Arn"],
        "scan_timestamp": utc_iso(),
        "region": sess.region_name or os.environ.get("AWS_DEFAULT_REGION") or "unknown"
    }

# IAM Users
def collect_iam(sess, cfg) -> List[Dict[str, Any]]:
    out = []
    iam = sess.client("iam", config=cfg)
    try:
        paginator = iam.get_paginator("list_users")
        for page in paginator.paginate():
            for u in page.get("Users", []):
                username = u.get("UserName")
                user_id  = u.get("UserId")
                arn      = u.get("Arn")
                create   = utc_iso(u.get("CreateDate"))
                last_activity = None
                try:
                    gu = iam.get_user(UserName=username)
                    last_activity = gu["User"].get("PasswordLastUsed")
                except ClientError:
                    pass
                policies = []
                try:
                    for ap in iam.get_paginator("list_attached_user_policies").paginate(UserName=username):
                        for p in ap.get("AttachedPolicies", []):
                            policies.append({"policy_name": p.get("PolicyName"), "policy_arn": p.get("PolicyArn")})
                except ClientError:
                    warn(f"Access denied listing policies for user {username}")
                out.append({
                    "username": username,
                    "user_id": user_id,
                    "arn": arn,
                    "create_date": create,
                    "last_activity": utc_iso(last_activity) if last_activity else None,
                    "attached_policies": policies
                })
    except ClientError as e:
        warn(f"Access denied for IAM operations - skipping user enumeration ({e.response['Error'].get('Code')})")
    return out

# EC2 Instance
def collect_ec2(sess, cfg) -> List[Dict[str, Any]]:
    ec2 = sess.client("ec2", config=cfg)
    items = []
    try:
        paginator = ec2.get_paginator("describe_instances")
        reservations = []
        for page in paginator.paginate():
            reservations.extend(page.get("Reservations", []))
        # collect AMI ids to resolve names
        ami_ids = set()
        for r in reservations:
            for inst in r.get("Instances", []):
                if "ImageId" in inst:
                    ami_ids.add(inst["ImageId"])
        ami_name = {}
        if ami_ids:
            try:
                for pg in ec2.get_paginator("describe_images").paginate(ImageIds=list(ami_ids)):
                    for img in pg.get("Images", []):
                        ami_name[img["ImageId"]] = img.get("Name")
            except ClientError:
                pass

        for r in reservations:
            for i in r.get("Instances", []):
                tags = {t["Key"]: t["Value"] for t in i.get("Tags", [])} if i.get("Tags") else {}
                sg_ids = [sg.get("GroupId") for sg in i.get("SecurityGroups", [])]
                items.append({
                    "instance_id": i.get("InstanceId"),
                    "instance_type": i.get("InstanceType"),
                    "state": i.get("State", {}).get("Name"),
                    "public_ip": i.get("PublicIpAddress"),
                    "private_ip": i.get("PrivateIpAddress"),
                    "availability_zone": i.get("Placement", {}).get("AvailabilityZone"),
                    "launch_time": utc_iso(i.get("LaunchTime")),
                    "ami_id": i.get("ImageId"),
                    "ami_name": ami_name.get(i.get("ImageId")),
                    "security_groups": sg_ids,
                    "tags": tags
                })
    except ClientError as e:
        warn(f"EC2 describe failed: {e.response['Error'].get('Code')}")
    return items

# S3 Buckets
def collect_s3(sess, cfg) -> List[Dict[str, Any]]:
    s3 = sess.client("s3", config=cfg)
    buckets = []
    try:
        resp = s3.list_buckets()
        for b in resp.get("Buckets", []):
            name = b["Name"]
            creation = utc_iso(b.get("CreationDate"))
            # region
            try:
                loc = s3.get_bucket_location(Bucket=name).get("LocationConstraint")
                region = loc or "us-east-1"
            except ClientError as e:
                warn(f"S3 get_bucket_location failed for {name}: {e.response['Error'].get('Code')}")
                region = None
            total = 0
            size = 0
            try:
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=name):
                    for obj in page.get("Contents", []) or []:
                        total += 1
                        size += int(obj.get("Size", 0))
            except ClientError as e:
                warn(f"S3 listing failed for {name}: {e.response['Error'].get('Code')}")
            buckets.append({
                "bucket_name": name,
                "creation_date": creation,
                "region": region,
                "object_count": total,
                "size_bytes": size
            })
    except ClientError as e:
        warn(f"S3 list_buckets failed: {e.response['Error'].get('Code')}")
    return buckets

#Security Groups
def collect_security_groups(sess, cfg) -> List[Dict[str, Any]]:
    ec2 = sess.client("ec2", config=cfg)
    groups = []
    try:
        paginator = ec2.get_paginator("describe_security_groups")
        for page in paginator.paginate():
            for g in page.get("SecurityGroups", []):
                def fmt_perm(p):
                    proto = p.get("IpProtocol", "all")
                    # ports
                    if proto in ("-1", "all"): pr = "all"
                    else:
                        f = p.get("FromPort"); t = p.get("ToPort")
                        pr = "all" if f is None or t is None else f"{f}-{t}"
                    srcs = [ip.get("CidrIp") for ip in p.get("IpRanges", [])] + [ip6.get("CidrIpv6") for ip6 in p.get("Ipv6Ranges", [])]
                    return {"protocol": "all" if proto == "-1" else proto, "port_range": pr, "source": ", ".join(srcs) or "-"}
                inbound = [fmt_perm(p) for p in g.get("IpPermissions", [])]
                outbound = [fmt_perm(p) for p in g.get("IpPermissionsEgress", [])]
                groups.append({
                    "group_id": g.get("GroupId"),
                    "group_name": g.get("GroupName"),
                    "description": g.get("Description"),
                    "vpc_id": g.get("VpcId"),
                    "inbound_rules": inbound,
                    "outbound_rules": outbound
                })
    except ClientError as e:
        warn(f"DescribeSecurityGroups failed: {e.response['Error'].get('Code')}")
    return groups

# Output format
def to_json(account: Dict[str,str], iam_users, ec2_instances, s3_buckets, sec_groups):
    return {
        "account_info": account,
        "resources": {
            "iam_users": iam_users,
            "ec2_instances": ec2_instances,
            "s3_buckets": s3_buckets,
            "security_groups": sec_groups
        },
        "summary": {
            "total_users": len(iam_users),
            "running_instances": sum(1 for i in ec2_instances if i.get("state") == "running"),
            "total_buckets": len(s3_buckets),
            "security_groups": len(sec_groups)
        }
    }
def print_table(account: Dict[str,str], iam_users, ec2_instances, s3_buckets, sec_groups):
    print(f"AWS Account: {account['account_id']} ({account['region']})")
    print(f"Scan Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    # IAM
    print(f"IAM USERS ({len(iam_users)} total)")
    print(f"{'Username':20} {'Create Date':12} {'Last Activity':12} Policies")
    for u in iam_users:
        last = (u['last_activity'] or '-')[:10]
        print(f"{u['username'][:20]:20} {u['create_date'][:10]:12} {last:12} {len(u['attached_policies'])}")
    print()
    # EC2
    running = sum(1 for i in ec2_instances if i.get("state") == "running")
    print(f"EC2 INSTANCES ({running} running, {len(ec2_instances)-running} stopped)")
    print(f"{'Instance ID':19} {'Type':10} {'State':9} {'Public IP':15} {'Launch Time':16}")
    for i in ec2_instances:
        print(f"{i['instance_id']:19} {i['instance_type'] or '-':10} {i['state'] or '-':9} {str(i['public_ip'] or '-'):15} {i['launch_time'][:16] if i['launch_time'] else '-':16}")
    print()
    # S3
    print(f"S3 BUCKETS ({len(s3_buckets)} total)")
    print(f"{'Bucket Name':28} {'Region':10} {'Created':12} {'Objects':7} {'Size (MB)':9}")
    for b in s3_buckets:
        mb = f"{(b['size_bytes']/1_000_000):.1f}" if b['size_bytes'] else "0.0"
        print(f"{b['bucket_name'][:28]:28} {str(b['region'] or '-'):10} {b['creation_date'][:10]:12} {b['object_count']:7} {mb:9}")
    print()
    # SG
    print(f"SECURITY GROUPS ({len(sec_groups)} total)")
    print(f"{'Group ID':14} {'Name':16} {'Inbound Rules'}")
    for g in sec_groups:
        print(f"{g['group_id']:14} {g['group_name'][:16]:16} {len(g['inbound_rules'])}")
    print()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", help="AWS region (default: from config)")
    ap.add_argument("--output", help="Write output to file (json only)")
    ap.add_argument("--format", choices=["json", "table"], default="json")
    args = ap.parse_args()
    try:
        sess, cfg = make_session(args.region)
	#verify authentication at startup using sts:GetCallerIdentity
        acct = verify_auth(sess)
    except (NoCredentialsError, NoRegionError, ClientError, EndpointConnectionError) as e:
        err(f"Authentication failed: {e}"); sys.exit(1)

    start = time.time()
    iam_users = collect_iam(sess, cfg)
    ec2_instances = collect_ec2(sess, cfg)
    s3_buckets = collect_s3(sess, cfg)
    sec_groups = collect_security_groups(sess, cfg)
    duration = time.time() - start

    if args.format == "json":
        payload = to_json(acct, iam_users, ec2_instances, s3_buckets, sec_groups)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        else:
            print(text)
    else:
        print_table(acct, iam_users, ec2_instances, s3_buckets, sec_groups)

if __name__ == "__main__":
    main()
