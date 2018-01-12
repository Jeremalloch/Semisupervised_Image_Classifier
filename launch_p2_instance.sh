#!/bin/bash
echo "Update this script to use a spot instance! Save your money"
aws ec2 run-instances --image-id ami-3b6bce43 --count 1 --instance-type p2.xlarge --key-name Jeremy --security-groups JupyterSecurityGroup

