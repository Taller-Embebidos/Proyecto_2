SUMMARY = "Sistema de semáforo inteligente con YOLO"
LICENSE = "CLOSED"

SRC_URI = "git://github.com/Taller-Embebidos/Proyecto_2.git;branch=main;protocol=https"
SRCREV = "${AUTOREV}"
S = "${WORKDIR}/git"

DEPENDS = "\
    python3-native \
    opencv \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
"

RDEPENDS:${PN} = "\
    python3 \
    python3-opencv \
    python3-numpy \
    tensorflow-lite \
    python3-ctypes \
    python3-json \
    bash \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-good-isomp4 \
"
do_install() {
    install -d ${D}${bindir}
    install -d ${D}${datadir}/semaforo

    cat > ${D}${bindir}/semaforo << 'SCRIPT'
#!/bin/bash
echo "========================================"
echo "    SEMÁFORO INTELIGENTE - YOLO TFLite"
echo "========================================"

# Configuración de display universal
export DISPLAY=:0
export XAUTHORITY=/home/root/.Xauthority

# Detectar si estamos en QEMU o hardware real
if [ -f /proc/device-tree/model ] && grep -q "QEMU" /proc/device-tree/model 2>/dev/null; then
    echo "Entorno: QEMU (emulación) - detectado por device-tree"
    ENV="qemu"
elif [ -f /etc/qemu-banner ]; then
    echo "Entorno: QEMU (emulación) - detectado por qemu-banner"
    ENV="qemu"
elif uname -a | grep -q "qemu"; then
    echo "Entorno: QEMU (emulación) - detectado por uname"
    ENV="qemu"
else
    echo "Entorno: Hardware real"
    ENV="rpi4"
fi

echo "Variable ENV establecida como: $ENV"

# Manejo de X11 según el entorno
if ! xset q >/dev/null 2>&1; then
    echo "X11 no está corriendo - intentando iniciar..."
    
    if [ "$ENV" = "qemu" ]; then
        echo "Iniciando X11 para QEMU..."
        X -nocursor :0 &
        sleep 5
    else
        echo "En hardware real, X11 debería iniciarse automáticamente"
        echo "Si no hay display, iniciando X11..."
        startx &
        sleep 3
    fi
    
    # Re-exportar variables después de iniciar X11
    export DISPLAY=:0
    export XAUTHORITY=/home/root/.Xauthority
    sleep 2
fi

# Verificar estado final de X11
if xset q >/dev/null 2>&1; then
    echo "✓ X11 funcionando en $DISPLAY"
else
    echo " X11 no disponible - modo headless"
fi

cd /usr/share/semaforo

echo "Ejecutando semáforo inteligente..."
# Forzar el entorno como variable de entorno
export SEMAFORO_ENVIRONMENT="$ENV"
python3 semaforo.py
SCRIPT
    chmod 0755 ${D}${bindir}/semaforo

    install -m 0755 ${S}/src/semaforo.py ${D}${datadir}/semaforo/
    install -m 0644 ${S}/src/yolo11n_float16.tflite ${D}${datadir}/semaforo/
    install -m 0644 ${S}/src/labels.txt ${D}${datadir}/semaforo/
    install -m 0644 ${S}/src/video_test.mp4 ${D}${datadir}/semaforo/
}


FILES:${PN} += "\
    ${bindir}/semaforo \
    ${datadir}/semaforo/semaforo.py \
    ${datadir}/semaforo/yolo11n_float16.tflite \
    ${datadir}/semaforo/labels.txt \
    ${datadir}/semaforo/video_test.mp4 \
"

INSANE_SKIP:${PN} += "already-stripped"
