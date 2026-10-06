#!/usr/bin/env bash
# One-time set-up of the rented server (Ubuntu 24.04 LTS), run as root over SSH (web demo
# item 17). It changes only this server. It doesn't buy, rent or pay for anything.
#
#   sudo bash deploy/setup-server.sh
#
# 1. A firewall that opens only SSH, 80 and 443.
# 2. Automatic security updates.
# 3. Docker and its compose plugin.
# 4. /etc/deixis/ for the key file (root only), and a reminder of how to fill it.
# Password login over SSH should already be off (key only): this checks and says.
set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "run as root: sudo bash $0" >&2; exit 1; }

echo "== firewall: SSH, 80, 443 only"
apt-get update -qq
apt-get install -y -qq ufw unattended-upgrades ca-certificates curl
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status verbose

echo "== automatic security updates"
dpkg-reconfigure -f noninteractive unattended-upgrades
systemctl enable --now unattended-upgrades

echo "== Docker"
if ! command -v docker >/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker
docker --version
docker compose version

echo "== the key's folder"
install -d -m 700 -o root -g root /etc/deixis
if [ ! -f /etc/deixis/webdemo.env ]; then
  install -m 600 -o root -g root /dev/null /etc/deixis/webdemo.env
  echo "Now put the key in it (never in the repo):  sudo nano /etc/deixis/webdemo.env"
  echo "  one line:  NEBIUS_API_KEY=..."
fi
ls -l /etc/deixis/

echo "== SSH: password login should be off"
if sshd -T 2>/dev/null | grep -qi '^passwordauthentication yes'; then
  echo "WARNING: password login is ON. Set 'PasswordAuthentication no' in /etc/ssh/sshd_config"
  echo "         (and in /etc/ssh/sshd_config.d/*.conf), then: systemctl restart ssh"
else
  echo "ok: password login is off"
fi
echo "done"
