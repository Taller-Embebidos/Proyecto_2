IMAGE_FEATURES += "ssh-server-dropbear x11-base"

IMAGE_INSTALL:append = " \
    semaforo-inteligente \
    tensorflow-lite \
    python3 \
    python3-numpy \
    python3-opencv \
    nano \
    xterm \
    xserver-xorg \
    xinit \
    mesa \
    matchbox-wm \
    matchbox-terminal \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-good-isomp4 \
    gtk+3 \
    networkmanager \
    networkmanager-nmtui \
    dhcpcd \
"

PACKAGECONFIG:append:pn-opencv = " gtk v4l"
