---
title: ubuntu配置
katex: true
typora-copy-images-to: ../../img/
date: 2021-10-08 09:52:47
updated: 2021-10-08 09:52:47
sticky: 0
tags: 
  - ubuntu
  - setup
  - linux
categories: Blog
toc: true
---

ubuntu常用设置.

<!-- more -->

## vim

```shell
sudo apt install vim

# 添加配置文件
vim ~/.vimrc

"set paste
"set nopaste
set expandtab
set softtabstop=2
set autoindent
set tabstop=2
set shiftwidth=2
set nu
syntax on
set mouse=a "支持鼠标滑轮
set mouse=v "支持鼠标选中复制
"set viminfo='1000,<500
```





## 用户

```bash
# 创建用户sanzo，指定home目录和登陆的shell
useradd -d /home/sanzo -s /bin/bash -m sanzo

# 设置登录密码
passwd sanzo

# 添加sudo组
usermod -a -G sudo sanzo

# 删除用户
sudo userdel -r test

# append ssh key to remote host
cat .ssh/id_ras.pub | ssh user@hostname "cat >> ~/.ssh/authorized_keys"
```





## 代理

> 配置v2ray

[v2ray-core/releases](https://github.com/v2fly/v2ray-core/releases)

```shell
mkdir v2ray && cd v2ray
wget https://raw.githubusercontent.com/v2fly/fhs-install-v2ray/master/install-release.sh
sudo bash install-release.sh --local ./v2ray-linux-64.zip

# local user run
nohup ./v2ray run config.json > v2ray.log 2>&1 &
```

[v2ray config.json配置](https://github.com/Sanzo00/files/blob/master/other/v2ray.json)

![image-20211008182055367](/img/Blog/linux/image-20211008182055367.png)



> proxychains

```shell
sudo apt install proxychains

# 修改配置文件
sudo vim /etc/proxychains.conf 
# 最后一行改为对应自己的端口
socks5    127.0.0.1 10800
```



> apt代理

```shell
sudo vim /etc/apt/apt.conf.d/proxy.conf
Acquire::http::Proxy "socks5h://127.0.0.1:10800";
Acquire::https::Proxy "socks5h://127.0.0.1:10800";
```



> bash代理

```shell
vim ~/.bashrc
# vim ~/.zshrc

export ALL_PROXY="socks5://127.0.0.1:10800"
export all_proxy="socks5://127.0.0.1:10800"
export http_proxy="http://127.0.0.1:10801"
export https_proxy="https://127.0.0.1:10801"
# export https_proxy="http://127.0.0.1:10801" # for conda http error
```

在setting中设置了http，apt和bash应该可以不用再设置了，以防万一可以加上。



## git

```shell
sudo apt install git

# 环境配置
git config --global user.email "arrangeman@163.com"
git config --global user.name "Sanzo00"

# 生成公钥和私钥
ssh-keygen -t rsa -C "your_email@example.com"

# 将公钥放到github中
cat ~/.ssh/id_rsa.pub 
```

> 代理

```bash
# http and https
git config --global http.proxy http://127.0.0.1:10801
git config --global https.proxy https://127.0.0.1:10801
# socks5
git config --global http.proxy socks5://127.0.0.1:10800
git config --global https.proxy socks5://127.0.0.1:10800

# unset
git config --global --unset http.proxy
git config --global --unset https.proxy


# ssh
sudo apt install connect-proxy
vim ~/.ssh/config
# socks5
Host github.com
User git
ProxyCommand connect -S 127.0.0.1:10800 %h %p
# http || https
Host github.com
User git
ProxyCommand connect -H 127.0.0.1:10801 %h %p
```





## zsh

https://sanzo.top/Linux/zsh/

```bash
# 安装zsh
sudo apt install zsh

# 安装ohmyzsh
sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# 切换shell为zsh
chsh -s /bin/zsh
```

 

> 插件

```bash
# 自动补全
git clone https://github.com/zsh-users/zsh-autosuggestions.git $ZSH_CUSTOM/plugins/zsh-autosuggestions

# 高亮
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting

# 修改配置文件
vim ~/.zshrc
plugins=(
  git
  zsh-autosuggestions
  zsh-syntax-highlighting
)

# 重置zsh环境
source ~/.zshrc
```



## 鼠标

> 修改滑轮速度

[IMWheel](https://wiki.archlinux.org/title/IMWheel#Run_IMWheel_on_startup_using_a_service)

```shell
sudo apt install imwheel
sudo vim ~/.imwheelrc

".*"
None,      Up,   Button4, 5 # 速度
None,      Down, Button5, 5 # 速度
Control_L, Up,   Control_L|Button4
Control_L, Down, Control_L|Button5
Shift_L,   Up,   Shift_L|Button4
Shift_L,   Down, Shift_L|Button5
```



> 开机自启

```shell
sudo ~/.config/systemd/user
vim ~/.config/systemd/user/imwheel.service

[Unit]
Description=IMWheel
Wants=display-manager.service
After=display-manager.service

[Service]
Type=simple
Environment=XAUTHORITY=%h/.Xauthority
ExecStart=/usr/bin/imwheel -d
ExecStop=/usr/bin/pkill imwheel
RemainAfterExit=yes

[Install]
WantedBy=graphical-session.target

systemctl --user daemon-reload
systemctl --user enable --now imwheel.service
journalctl --user --unit imwheel.service
```





## 网速监控

```shell
sudo add-apt-repository ppa:fossfreedom/indicator-sysmonitor -y
sudo apt update
sudo apt-get install indicator-sysmonitor

# 运行
indicator-sysmonitor

cpu: ({cpu} {cputemp}) gpu: ({nvgputemp}) mem: ({mem}) net: {net} {totalnet}
```

![image-20211008210121275](/img/Blog/linux/image-20211008210121275.png)

![image-20211008205948437](/img/Blog/linux/image-20211008205948437.png)



## 截图

我在windows上用的[snipaste](https://www.snipaste.com/)，不过linux还没出，有两个方案可以代替：

1、系统默认的截图工具

- `PrtSc` – 获取整个屏幕的截图并保存到 Pictures 目录。
- `Shift + PrtSc` – 获取屏幕的某个区域截图并保存到 Pictures 目录。
- `Alt + PrtSc` –获取当前窗口的截图并保存到 Pictures 目录。
- `Ctrl + PrtSc` – 获取整个屏幕的截图并存放到剪贴板。
- `Shift + Ctrl + PrtSc` – 获取屏幕的某个区域截图并存放到剪贴板。
- `Ctrl + Alt + PrtSc` – 获取当前窗口的 截图并存放到剪贴板。

2、[flameshot](https://github.com/flameshot-org/flameshot)

```shell
# install
apt install flameshot

# start
flameshot gui
```



## 显卡

[显卡算力](https://developer.nvidia.com/zh-cn/cuda-gpus#compute)

[cuda对应的最低驱动版本](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

[Ubuntu20.04安装NVIDIA显卡驱动+cuda+cudnn配置深度学习环境](https://www.mlzhilu.com/archives/ubuntu2004%E5%AE%89%E8%A3%85nvidia%E6%98%BE%E5%8D%A1%E9%A9%B1%E5%8A%A8)

> 驱动安装

```shell
# 查看显卡型号
lspci | grep -i nvidia
```

![image-20211008193000787](/img/Blog/linux/image-20211008193000787.png)

[下载驱动](https://www.nvidia.com/Download/index.aspx?lang=en-us)

![image-20211008193044041](/img/Blog/linux/image-20211008193044041.png)

```shell
sudo apt install -y lightdm gcc make
sudo passwd root

# 切换桌面，选择lightdm
sudo dpkg-reconfigure gdm3

# 关闭lightdm桌面
systemctl stop lightdm

sudo chmod a+x NVIDIA-Linux-x86_64-450.80.02.run
sudo ./NVIDIA-Linux-x86_64-450.80.02.run -no-x-check -no-nouveau-check -no-opengl-files --no-kernel-module
# -no-x-check:安装时关闭X服务
# -no-nouveau-check: 安装时禁用nouveau
# -no-opengl-files:只安装驱动文件，不安装OpenGL文件
# 后面出来的提示，选择默认选项
```

```shell
# 测试是否成功
nvidia-smi
```

![image-20211008204151165](/img/Blog/linux/image-20211008204151165.png)





如果出现`/dev/xxx: clean`的问题，进不了桌面，可能是因为驱动不匹配的问题。可以删除`/etc/X11/xorg.conf`。



如果出现`An NVIDIA kernel module 'nvidia-drm' appears to already be loaded in your kernel. This may be because it is in use (for example, by an X server, a CUDA program, or the NVIDIA Persistence Daemon), but this may also happen if your kernel was configured without support for module unloading.`

```bash
# https://unix.stackexchange.com/questions/440840/how-to-unload-kernel-module-nvidia-drm

sudo systemctl isolate multi-user.target
modprobe -r nvidia-drm

# reinstall you nvidia driver
# then
sudo systemctl start graphical.target
```





> 安装cuda

[下载cuda](https://developer.nvidia.com/cuda-toolkit-archive)，这里我选择的是cuda 11.2。

![image-20211008204125042](/img/Blog/linux/image-20211008204125042.png)

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
sudo sh cuda_11.2.0_460.27.04_linux.run
```



回车取消勾选`Driver`，因为前面已经装过驱动，然后install，也可以在Options中自定义安装位置。

![image-20211008204241869](/img/Blog/linux/image-20211008204241869.png)

在.bashrc文件中配置环境变量

```shell
export PATH=/usr/local/cuda-11.2/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH}
```

```shell
source ～/.bashrc

# 测试cuda
nvcc -V
```

![image-20211008204439775](/img/Blog/linux/image-20211008204439775.png)



> 安装cudnn

[下载cudnn](https://developer.nvidia.com/rdp/cudnn-download)

```shell
# 将文件复制到cuda对应的文件夹下
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/include/cudnn_version.h /usr/local/cuda/include/ # for cudnn v8+
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
```

> 测试

在`~/NVIDIA_CUDA-11.2_Samples`下编译代码，然后运行cuda提供的例子。

![image-20211008204814707](/img/Blog/linux/image-20211008204814707.png)



![image-20211008204917470](/img/Blog/linux/image-20211008204917470.png)



> 踩坑

1. apt 更新软件包之后，导致显卡驱动失效，这是因为内核版本发生了变化，将内核回退到上一个版本即可。

   https://www.jianshu.com/p/5ac8aa703075

   

## 显示器

适用于多个显示器。

我这里有两块屏幕`HDMI-1`，`HDMI-1-0`。

```shell
# 查看当前显示器信息
xrandr

# 设置HDMI-1为主屏幕
xrandr --output HDMI-1 --primary

# 只显示一个屏幕，关闭HDMI-1-0屏幕
xrandr --output HDMI-1 --auto --output HDMI-1-0 --off

# 复制屏幕
xrandr --output HDMI-1-0 --same-as HDMI-1 --auto

# 设置HDMI-1-0为左扩展屏
xrandr --output HDMI-1-0 --left-of HDMI-1 --auto

# 设置HDMI-1-0右扩展屏
xrandr --output HDMI-1-0 --right-of HDMI-1 --auto
```

> 开机自启

还没找到合适的开机自启命令，不过可以在系统设置中调。



## VSCode

[vscode](https://code.visualstudio.com/)



> server端网络不好，“Downloading VS Code Server”

进入`~/.vscode-server/bin/`查看vscode server的commit id，然后手动下载上传到服务器。

下载链接：`https://update.code.visualstudio.com/commit:${commit_id}/server-linux-x64/stable`

解压之后重命名为commit id，然后放到`~/.vscode-server/bin/`





## 其他

[typora](https://typora.io/#linux)



[python环境](https://sanzo.top/Default/python-env/?highlight=python)



<!-- Q.E.D. -->

