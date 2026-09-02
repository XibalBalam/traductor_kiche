#!/bin/bash
# Script para configurar un servidor AWS EC2 (Ubuntu) en la Capa Gratuita (t2.micro - 1GB RAM)
# Este script crea Memoria Virtual (Swap) para que los modelos de IA no colapsen el servidor por falta de RAM.

echo "======================================="
echo " Configurando VPS Gratuito (AWS EC2)   "
echo "======================================="

# 1. Crear 4GB de Memoria Virtual (Swap) para evitar OOM (Out of Memory)
echo "[1/5] Configurando 4GB de Memoria Swap (esencial para la capa gratuita)..."
if grep -q "swapfile" /etc/fstab; then
    echo "El archivo swap ya existe."
else
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    # Optimizar el uso del swap
    sudo sysctl vm.swappiness=10
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
    echo "Memoria Swap configurada con éxito."
fi

# 2. Actualizar sistema e instalar dependencias necesarias
echo "[2/5] Instalando dependencias del sistema (Nginx, Python, Certbot)..."
sudo apt update -y
sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx ffmpeg tmux

# 3. Preparar el entorno de Python
echo "[3/5] Preparando entorno de Python..."
cd /home/ubuntu/traductorKiche-Esp/traductor_kiche || exit
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configurar Nginx para el dominio
echo "[4/5] Configurando Nginx para el dominio tradkiche.zentary.net..."
cat <<EOF | sudo tee /etc/nginx/sites-available/traductor
server {
    listen 80;
    server_name tradkiche.zentary.net;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_addrs;
        
        # Aumentar timeout por si el modelo tarda en responder debido al swap
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/traductor /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo systemctl restart nginx

# 5. Instrucciones Finales
echo "======================================="
echo "✅ Servidor configurado correctamente."
echo ""
echo "PASOS FINALES:"
echo "1. Ve a donde compraste tu dominio (zentary.net) y crea un registro 'A' apuntando 'tradkiche' a la IP Pública de este servidor."
echo "2. Luego, ejecuta el siguiente comando para activar HTTPS (Gratis):"
echo "   sudo certbot --nginx -d tradkiche.zentary.net"
echo "3. Finalmente, para correr la app usa tmux:"
echo "   tmux"
echo "   source venv/bin/activate"
echo "   gunicorn -c gunicorn_config.py"
echo "======================================="
