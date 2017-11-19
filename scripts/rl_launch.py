import datetime
import json
import os
import tempfile

import click

AMI_MAP = {
    "us-east-1": "xxx",
}
CODE_URL = {
    'rllabsharp':'xx',
    'mergerllab': 'xxx'
}

def highlight(x):
    if not isinstance(x, str):
        x = json.dumps(x, sort_keys=True, indent=2)
    click.secho(x, fg='green')


def upload_archive(exp_name, archive_excludes, s3_bucket):
    import hashlib, os.path as osp, subprocess, tempfile, uuid, sys 

    # Archive this package
    thisfile_dir = osp.dirname(osp.abspath(__file__))
    pkg_parent_dir = osp.abspath(osp.join(thisfile_dir, '..', '..'))
    pkg_subdir = osp.basename(osp.abspath(osp.join(thisfile_dir, '..')))
    
    assert osp.abspath(__file__) == osp.join(pkg_parent_dir, pkg_subdir, "scripts", "rl_launch.py")

    # run tar
    tmpdir = tempfile.TemporaryDirectory()
    local_archive_path = osp.join(tmpdir.name, '{}.tar.gz'.format(uuid.uuid4))
    tar_cmd = ["tar", "-zcvf" , local_archive_path, "-C", pkg_parent_dir]
    for pattern in archive_excludes:
        tar_cmd += ['--exclude', pattern]
    tar_cmd += ["-h", pkg_subdir]
    print(tar_cmd)

    if sys.platform == 'darwin':
        # Prevent Mac tar from adding ._* files
        env = os.environ.copy()
        env['COPYFILE_DISABLE'] = '1'
        subprocess.check_call(tar_cmd, env=env)
    else:
        subprocess(tar_cmd)
    
    # Construct remote path to place the archive on S3
    with open(local_archive_path, 'rb') as f:
        archive_hash = hashlib.sha224(f.read()).hexdigest()
    remote_archive_path = '{}/{}_{}.tar.gz'.format(s3_bucket.strip(), exp_name.strip(), archive_hash.strip())

    # Upload
    upload_cmd = ["aws", "s3", "cp", local_archive_path, remote_archive_path]
    highlight(" ".join(upload_cmd))
    subprocess.check_call(upload_cmd)

    presign_cmd = ["aws", "s3", "presign", remote_archive_path, "--expires-in", str(60 * 60 * 24 * 30)]
    highlight(" ".join(presign_cmd))
    remote_url = subprocess.check_output(presign_cmd).decode("utf-8").strip()
    return remote_url


def json_to_flags(exp_json):
    flag_str = '  '
    log_dir = ''
    for key, value in exp_json.items():
        flag_str += '--'+str(key) + ' ' + str(value)+ ' '
        log_dir += str(key) + '=' + str(value)+'-'
    return flag_str, log_dir


# todo check if the rllab url is right
def make_download_and_run_script(code_url, cmd):
    return """su -l ubuntu << 'EOF'
set  -x
cd ~
wget "{code_url}" -O code.tar.gz
tar xvaf code.tar.gz
rm code.tar.gz
export PATH="/opt/anaconda/bin:$PATH"
export PYTHONPATH=/home/ubuntu/rllabsharp:$PYTHONPATH
cd rllabsharp/sandbox/rocky/tf/launchers/
{cmd}
EOF
""".format(code_url=code_url, cmd=cmd)



# TODO finish up this part
def make_run_script(code_url, exp_str):
    cmd = """
python algo_gym_stub.py {exp_str}
""".format(exp_str=exp_str)

    return """#!/bin/bash
{
set -x

%s

} >> /home/ubuntu/user_data.log 2>&1
""" %(make_download_and_run_script(code_url, cmd))


@click.command()
@click.argument('exp_files', nargs=-1, type=click.Path(), required=True)
@click.option('--key_name', default=lambda: os.environ["KEY_NAME"])
@click.option('--aws_access_key_id', default=os.environ.get("AWS_ACCESS_KEY", None))
@click.option('--aws_secret_access_key', default=os.environ.get("AWS_ACCESS_SECRET", None))
@click.option('--archive_excludes', default=(".git", "__pycache__", "haoliu_temporary", ".idea", "scratch", '.pyc', '.vscode', 'exp_log', 'exp_logs', 'data'))
@click.option('--s3_bucket', default=os.environ.get("RLLAB_S3_BUCKET", None))
@click.option('--region_name',default="us-east-1")
@click.option('--zone',default="us-east-1a")
@click.option('--cluster_size', type=int, default=1)
@click.option('--instance_type', default='c4.large')
@click.option('--security_group', default='default')
@click.option('--autoscale', is_flag=False, help='Use for autoscale')
@click.option('--yes', is_flag=True, help='Skip confirmation prompt')
def main(exp_files,
        key_name,
        aws_access_key_id,
        aws_secret_access_key,
        archive_excludes,
        s3_bucket,
        region_name,
        zone,
        cluster_size,
        instance_type, 
        security_group,
        autoscale,
        yes):
    
    highlight('Launching:')
    highlight(locals())

    import boto3
    ec2 = boto3.resource(
        "ec2",
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    as_client = boto3.client(
        'autoscaling',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


    for i_exp_file, exp_file in enumerate(exp_files):
        print(exp_file)
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
        highlight('Experiment [{}/{}]:'.format(i_exp_file + 1, len(exp_files)))
        highlight(exp)
        # if not yes:
        #     click.confirm('Coninoue?', abort=True)
        
        exp_prefix = exp['exp_prefix']
        env_name = exp['env_name']
        #exp_str = json.dum(exp)
        exp_str, log_flags = json_to_flags(exp)
        print('exp_str here.............', exp_str)
        exp_str += ' --log_dir '+ str(log_flags)
        exp_name = '{}_{}'.format(exp_prefix, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        print(exp_name)
        if exp_prefix.startswith('rllab'):
            code_url = CODE_URL['rllabsharp'] # upload_archive(exp_name, archive_excludes, s3_bucket)
        elif exp_prefix.startswith('mergerllab'):
            code_url = CODE_URL['mergerllab']
        else:
            raise NotImplementedError
        print(code_url)

        image_id = AMI_MAP[region_name]
        highlight('Using AMI: {}'.format(image_id))

        init_command = make_run_script(code_url, exp_str)
        print(init_command)
        # TODO: check if spot instance necessary here

        import base64 
        # create instance
        instance = ec2.create_instances(
            ImageId = image_id,
            KeyName=key_name,
            InstanceType=instance_type,
            EbsOptimized=True,
            SecurityGroups=[security_group],
            MinCount=1,
            MaxCount=1,
            Placement=dict(
                AvailabilityZone=zone,
            ),
            UserData=make_run_script(code_url, exp_str) # base64.b64encode(make_run_script(code_url, exp_str).encode()).decode()
            )[0]

        # TODO: add tags that fit our experiments
        instance.create_tags(
            Tags=[
                dict(Key='owner', Value='mergerllab'),
                dict(Key="exp_prefix", Value=exp_prefix),
                dict(Key="Name", Value=exp_name),
                dict(Key='env_name', Value=env_name)
            ]
        )
        highlight("Instance created. IP: %s"% instance.public_ip_address)

        if autoscale:
            asg_resp = as_client.create_auto_scaling_group(
                AutoScalingGroupName=exp_name,
                LaunchConfigurationName=exp_name,
                MinSize=cluster_size,
                MaxSize=cluster_size,
                DesiredCapacity=cluster_size,
                AvailabilityZones=[zone],
                Tags=[
                    dict(Key="Name", Value=exp_name + "-worker"),
                    dict(Key="es_dist_role", Value="worker"),
                    dict(Key="exp_prefix", Value=exp_prefix),
                    dict(Key="exp_name", Value=exp_name),
                ]
                # todo: also try placement group to see if there is increased networking performance
            )
            assert asg_resp["ResponseMetadata"]["HTTPStatusCode"] == 200
            highlight("Scaling group created")

        highlight("%s launched successfully." %exp_name)
        highlight("Manage at %s" % (
            "https://%s.console.aws.amazon.com/ec2/v2/home?region=%s#Instances:sort=tag:Name" % (
            region_name, region_name)
        ))
    


if __name__ == '__main__':
    main()

















