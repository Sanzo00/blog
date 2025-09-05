---
title: Jump the Wall
hidden: false
katex: true
sticky: 0
typora-copy-images-to: ../../img
date: 2022-05-19 11:13:08
updated: 2022-05-19 11:13:08
tags: v2ray
categories: Blog
toc: true
---



科学上网

<!-- more -->

---



## V2ray



### 服务端

测试ip：[ping.chinaz.com](https://ping.chinaz.com)

证书申请：https://freessl.cn/

```bash
#X-ui面板安装
bash <(curl -Ls https://raw.githubusercontent.com/vaxilu/x-ui/master/install.sh)


# ufw开启端口
ufw allow 40000:60000/tcp
ufw allow 40000:60000/udp
```



### 客户端

Android，linux，macOS 安装包：https://github.com/v2fly/v2ray-core/releases

v2ray客户端配置文件：https://github.com/Sanzo00/files/blob/master/other/v2ray.json

linux端下载v2ray安装包之后，可以选择安装到本地或者直接运行可执行文件。

> 安装到本地

```bash
wget https://raw.githubusercontent.com/v2fly/fhs-install-v2ray/master/install-release.sh
# 自动选择安装包
sudo bash install-release.sh

# 指定安装包
sudo bash install-release.sh --local ./v2ray-linux-64.zip

# 将配置文件拷贝到 /usr/local/etc/v2ray/config.json
```



> 直接运行可执行文件

```bash
nohup ./v2ray run config.json > v2ray.log 2>&1 &
```



## Clash

> 开源桌面应用

Mac：[ClashX](https://github.com/yichengchen/clashX)

Windows：[ClashForWindows](https://github.com/Fndroid/clash_for_windows_pkg/releases)

Android：[ClashForAndroid](https://github.com/Kr328/ClashForAndroid)





> 服务器配置

下载对应系统的可执行文件：[Dreamacro/clash  release](https://github.com/Dreamacro/clash/releases)

V2ray to Clash节点转换工具：[v2rayse.com/node-convert](https://v2rayse.com/node-convert)



```bash
cd ~/software
mkdir clash && cd clash

# 下载软件包
wget https://github.com/Dreamacro/clash/releases/download/v1.13.0/clash-linux-amd64-v1.13.0.gz

# 解压并添加权限
gunzip clash-linux-amd64-v1.13.0.gz
mv clash-linux-amd64-v1.13.0 clash
chmod +x clash

# 下载Country.mmdb
wget https://github.com/Dreamacro/maxmind-geoip/releases/download/20230312/Country.mmdb

# 设置clash配置文件
vim config.yaml

```



创建systemd配置文件

```bash
sudo vim /etc/systemd/system/clash.service
```



```bash
[Unit]
Description=Clash daemon, A rule-based proxy in Go.
After=network.target

[Service]
Type=simple
Restart=always
ExecStart=/home/sanzo/software/clash/clash -d /home/sanzo/software/clash/

[Install]
WantedBy=multi-user.target
```

`/home/sanzo/software/clash/clash` is you clash executable file

`/home/sanzo/software/clash/` is your clash config directory



使用systemctl控制clash的运行：

```bash
sudo systemctl enable clash
sudo systemctl status clash
sudo systemctl start clash
sudo systemctl restart clash
sudo systemctl stop clash
```



网络测试

```bash
wget google.com
```




## 设置终端代理

### 终端代理

```bash
vim ~/.bashrc

export ALL_PROXY="socks5://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```



### git代理

> 对http和https代理

```bash
# http and https
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy https://127.0.0.1:7890

# socks5
git config --global http.proxy socks5://127.0.0.1:7890
git config --global https.proxy socks5://127.0.0.1:7890

# unset
git config --global --unset http.proxy
git config --global --unset https.proxy
```



> 对ssh代理：

```bash
sudo apt install connect-proxy
# (or) https://github.com/larryhou/connect-proxy
vim ~/.ssh/config

# socks5
Host github.com
User git
ProxyCommand connect -S 127.0.0.1:7890 %h %p

# http || https
Host github.com
User git
ProxyCommand connect -H 127.0.0.1:7890 %h %p
```





<!-- Q.E.D. -->
