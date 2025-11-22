IMAGE_FEATURES += "ssh-server-dropbear x11-base"

IMAGE_INSTALL:append = " \
    semaforo-inteligente \
    python3 \
    python3-opencv \
    python3-numpy \
    tensorflow-lite \
    libgomp \
    libstdc++ \
    nano \
    xterm \
    xserver-xorg \
    xhost \
    mesa \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
"
