# Deshabilita el display manager y habilita la consola tty1
SYSTEMD_AUTO_ENABLE:remove = "display-manager.service"
SYSTEMD_AUTO_ENABLE:append = " getty@tty1.service"

IMAGE_FEATURES += "x11-base"

# Paquetes adicionales a instalar
IMAGE_INSTALL:append = " \
    semaforo-inteligente \
    python3-opencv \
    python3-numpy \
    tensorflow-lite \
    v4l-utils \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    xserver-xorg \
    mesa \
    mesa-driver-vc4 \
"

# Rutas extras para archivos locales
FILESEXTRAPATHS:prepend := "${THISDIR}/files:"

SRC_URI:append = " file://cmdline.txt"
