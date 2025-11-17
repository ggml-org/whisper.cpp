# 🧠 Preparing Whisper CPP WebSocket Server

This guide explains how to set up a **Whisper CPP WebSocket Server** on a fresh **Ubuntu 22.04** machine with **1× L4 GPU** or **1× A10 GPU**.

---

## 🧩 Step 1: Install NVIDIA Drivers & Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
sudo mkdir -p /usr/share/keyrings

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

sudo chmod a+r /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo apt-get install -y libwebsockets-dev libsdl2-dev cmake build-essential
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y nlohmann-json3-dev
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

If you see GPU info, the setup is successful. ✅

---

## 📦 Step 2: Clone and Prepare the Repository

**Assume your working directory is `/home/ubuntu`**

```bash
cd /home/ubuntu
git clone https://github.com/DataQueue-AI/whisper.cpp
```

---

## 🧰 Step 3: Add Arabic Model File

```bash
cd /home/ubuntu/whisper.cpp/models
```

Copy your **Whisper model `.bin` file** into that folder.

---

## 🎧 WS Streaming API

### 🔹 Endpoints

| Method | Endpoint  | Description                  |
| :----- | :-------- | :--------------------------- |
| `WS`   | `/stream` | WebSocket streaming endpoint |

---

### 🧪 Test with Client

Edit `ws_client.py` and set your server IP address, then run:

```bash
python3 ws_client.py
```

You can change the number of clients and chunk size:

```python
CONCURRENT_CLIENTS = 1
CHUNK_SECONDS = 0.2
```

---

## 🏗️ Step 4: Build Whisper CPP

```bash
cd /home/ubuntu/whisper.cpp
cmake -B build -DGGML_CUDA=ON -DWHISPER_SDL2=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build -j --config Release
```

### ✅ CUDA Architectures
| GPU | Architecture |
|------|-------------|
| A10 | `86` |
| L4  | `89` |

---

## 🧰 Step 5: Add systemd Service

Create file:

```
/etc/systemd/system/whisper-ws@.service
```

Content:

```ini
[Unit]
Description=Whisper.cpp WebSocket Server on port %i
After=network.target

[Service]
User=root
Environment="LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64"

ExecStart=/home/ubuntu/whisper.cpp/build/bin/whisper-ws \
    -m /home/ubuntu/whisper.cpp/models/ASR_models_streamingmodel_whisper.cpp_quantized_O5Arzspfull_streamsplits_7k-ggml-model-q5_0.bin \
    --port %i \
    -t 4 \
    -l ar \
    --step 400

WorkingDirectory=/home/ubuntu/whisper.cpp/build/bin
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Notes:
- Replace the model path with your actual model file path.
- Check if `/usr/local/cuda-13.0/lib64` exists (if not, install CUDA 13.0).
- `--step` defines chunk size in ms.

Reload services:

```bash
sudo systemctl daemon-reload
```

### Run one worker:

```bash
sudo systemctl enable whisper-ws@9001.service
sudo systemctl start whisper-ws@9001.service
sudo systemctl status whisper-ws@9001.service
```

### Run multiple workers (example: 8 workers):

```bash
for i in {9001..9008}; do
    sudo systemctl enable whisper-ws@$i
    sudo systemctl start whisper-ws@$i
done

sudo systemctl status 'whisper-ws@*'
```

---

## 🧰 Step 6 (Optional): Install CUDA 13

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
rm cuda-keyring_1.1-1_all.deb

sudo apt-get install -y cuda-toolkit-13-0
sudo reboot
```

After reboot:

```bash
which nvcc
```

It should show something like:

```
/usr/local/cuda-13.0/bin/nvcc
```

---

## 🧰 Step 7: Install NGINX

```bash
sudo apt-get update
sudo apt-get install -y nginx
sudo rm /etc/nginx/sites-enabled/default
sudo nano /etc/nginx/sites-available/whisper
```

---

### NGINX (One Worker)

```nginx
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location /stream {
        proxy_pass http://127.0.0.1:9001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
```

---

### NGINX (Multiple Workers — Example: 8)

```nginx
upstream whisper_ws_backends {
    least_conn;
    server 127.0.0.1:9001;
    server 127.0.0.1:9002;
    server 127.0.0.1:9003;
    server 127.0.0.1:9004;
    server 127.0.0.1:9005;
    server 127.0.0.1:9006;
    server 127.0.0.1:9007;
    server 127.0.0.1:9008;
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location /stream {
        proxy_pass http://whisper_ws_backends;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
```

Enable:

```bash
sudo ln -s /etc/nginx/sites-available/whisper /etc/nginx/sites-enabled/whisper
sudo nginx -t && sudo systemctl reload nginx
```

---

> 💡 **Tip:** Keep the model checkpoint on a fast NVMe SSD for optimal inference performance.

