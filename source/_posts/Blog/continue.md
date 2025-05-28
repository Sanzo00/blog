---
title: 'VSCode插件: Continue'
katex: true
sticky: 0
toc: true
typora-copy-images-to: ../../img/continue
date: 2024-04-02 19:23:09
updated: 2024-04-02 19:23:09
tags:
- continue
- ollama
categories:
- Blog
---



让大模型帮你写代码：[Ollama](https://ollama.com/download) + [Continue](https://github.com/continuedev/continue)。

<!-- more -->



## Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

我是在服务器上安装的，为了支持远程访问，需要修改下配置文件。

```bash
sudo vim  /etc/systemd/system/ollama.service


[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=root
Group=root
Restart=always
RestartSec=3
Environment="OLLAMA_MODELS=your_model_path/ollama/models"
# 添加下面两行
Environment="OLLAMA_HOST=0.0.0.0"
Environment="OLLAMA_ORIGINS=*"
Environment="CUDA_VISIBLE_DEVICES=0,1,2"

[Install]
WantedBy=default.target
```



ollama所有可用模型：https://ollama.com/library

```bash
# 重启服务
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 下载模型
ollama run codellama:13b
```



同时部署多个模型 **[可选]**

```bash
# 默认端口为11434， 这里额外部署一个11435的服务
OLLAMA_HOST=0.0.0.0:11435 ollama serve

# 下载模型
OLLAMA_HOST=127.0.0.1:11435 ollama pull codellama:13b

# 测试
curl -X POST http://127.0.0.1:11435/api/generate -d '{
  "model": "codellama:13b",
  "prompt": "Write me a function that outputs the fibonacci sequence"
}'
```







## Continue

1. 在vscode的插件库上安装Continue插件。

2. 在`config.json`中添加模型配置文件。

   如果使用本地模型，可以注释掉`"apiBase": "http://your_server_ip:11435"`，如果使用的是默认的ollama服务，端口`11435` 改为 `11434`。

   ```bash
   "models": [
       {
         "title": "Codellama 7b",
         "provider": "ollama",
         "model": "codellama:7b",
         "apiBase": "http://your_server_ip:11435"
       },
       {
         "title": "Codellama 13b",
         "provider": "ollama",
         "model": "codellama:13b",
         "apiBase": "http://your_server_ip:11435"
       },
       {
         "title": "Codellama 34b",
         "provider": "ollama",
         "model": "codellama:34b",
         "apiBase": "http://your_server_ip:11435"
       },
       {
         "title": "StarCoder2 3b",
         "provider": "ollama",
         "model": "starcoder2:3b",
         "apiBase": "http://node1:11435"
       },
       {
         "title": "StarCoder2 7b",
         "provider": "ollama",
         "model": "starcoder2:7b",
         "apiBase": "http://node1:11435"
       },
       {
         "title": "starcoder2:15b",
         "provider": "ollama",
         "model": "starcoder2:15b",
         "apiBase": "http://node1:11435"
       },
       {
         "title": "Llama2 7b",
         "provider": "ollama",
         "model": "llama2:7b",
         "apiBase": "http://your_server_ip:11435"
       },
       {
         "title": "Llama2 13b",
         "provider": "ollama",
         "model": "llama2:13b",
         "apiBase": "http://your_server_ip:11435"
       },
       {
         "title": "Llama2 70b",
         "provider": "ollama",
         "model": "llama2:70b",
         "apiBase": "http://your_server_ip:11435"
       }
     ],
   ```

   

## 效果

1. `Ctrl + L`选中代码，弹出聊天窗口，可以直接询问LLM，例如编写单元测试，检查bug。

   ![image-20240402195237473](/img/continue/image-20240402195031626.png)

2. `Ctrl + I`插入代码，弹出`prompt`输入框，根据需求生成代码。

   ![image-20240402195237473](/img/continue/image-20240402195237473.png)



![image-20240402195249229](/img/continue/image-20240402195249229.png)
