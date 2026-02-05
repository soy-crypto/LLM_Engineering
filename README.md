# LLM_Engineering


################################
Pod configuration.
GPU: RTX4090
Container image: nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
Container Start Command:

bash -c 'apt update; \
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y; \
mkdir -p ~/.ssh; \
cd ~/.ssh; \
chmod 700 ~/.ssh; \
echo "$PUBLIC_KEY" >> authorized_keys; \
chmod 700 authorized_keys; \
service ssh start; \
sleep infinity'

Container disk: 200GB Volume disk 200GB Volume mount.

####################
How to use ssh and sftp
step1:
SSH key genarate:
ssh-keygen -t ed25519 -C "tzhu2@scu.edu"
step2: rename ed25519 to ed25519.ppk (OS is windows)
step3: open SCP, host: 157.157.221.29 port: 30527 username: root. no need password, as we have private key in our machine.
step4: connect the host.

'''