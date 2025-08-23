FROM agile4im/cybergod_worker_terminal

COPY <<EOF /etc/apt/sources.list
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse

# 以下安全更新软件源包含了官方源与镜像站配置，如有需要可自行修改注释切换
deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
# deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
# # deb-src http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-proposed main restricted universe multiverse
EOF

RUN apt-get update
RUN apt-get install -y unzip
RUN apt-get install -y curl
RUN apt-get install -y vim
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip install vimgolf-gym==0.0.2

RUN mkdir -p /root/.cache/cybergod-vimgolf-dataset
RUN curl -Lk -o /root/.cache/cybergod-vimgolf-dataset/challenges.zip http://www.kaggle.com/api/v1/datasets/download/jessysisca/vimgolf-challenges-and-solutions
RUN unzip /root/.cache/cybergod-vimgolf-dataset/challenges.zip -d /root/.cache/cybergod-vimgolf-dataset
RUN rm /root/.cache/cybergod-vimgolf-dataset/challenges.zip
RUN touch /root/.cache/cybergod-vimgolf-dataset/challenges/DATASET_DOWNLOADED
