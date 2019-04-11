# cadCAD demo
Server to run webpage for basic cadCAD demo.

Get Docker from Canvas

To build Docker image:
```bash
sudo docker build -t blockscience/cadCADDemo:latest .
```
To run Docker image:
```bash
sudo docker run -d --restart=always -p 80:80 blockscience/cadCADDemo
```
Remove old images:
```bash
sudo docker system prune -f
```
Installing Docker:
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04


```bash
#!/bin/bash
# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
apt-cache policy docker-ce
sudo apt-get install -y docker-ce
sudo systemctl status docker
# Download and run image
sudo docker login -u <username> -p <password>
sudo docker run run -d --restart=always -p 80:80 aclarkdata/scribe:latest
```
