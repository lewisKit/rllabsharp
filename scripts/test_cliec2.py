#! /usr/bin/env python

import multiprocessing
import os
import sys 

import boto3



aws_access_key_id = os.environ.get("AWS_ACCESS_KEY", None)
aws_secret_access_key = os.environ.get("AWS_ACCESS_SECRET", None)
REGIONS = [
    "us-east-1",
]



def highlight(x):
    if not isinstance(x, str):
        x = json.dumps(x, sort_keys=True, inden=2)
    
    click.secho(x, fg='green')

def _collect_instances(region):
    client = boto3.client(
        "ec2",
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    
    print("Collecting instances in region", region)

    instances = [x['Instances'][0] for x in client.describe_instances(
        Filters=[
            {
                'Name': 'instance-state-name',
                'Values': [
                    'running'
                ]
            },
        ]
    )['Reservations']]

    for instance in instances:
        print(instance)
        # print(instance['InstanceId'], instance['PublicIpAddress'])


if __name__ == '__main__':
    # _collect_instances(REGIONS[0])
    ec2 = boto3.resource('ec2')
    instances = ec2.instances.filter(
        Filters=[
            {
                "Name": "instance-state-name", 
                "Values": ['running'], 
                
            },
            {
                "Name": "instance-type",
                "Values": ['c4.2xlarge']
            }]
    )

    for instance in instances:
        print(instance.id, instance.instance_type)



