
SUMMARY = "Sistema de semáforo inteligente con YOLO"
LICENSE = "CLOSED"

SRC_URI = "git://github.com/Taller-Embebidos/Proyecto_2.git;branch=main;protocol=https"
SRCREV = "${AUTOREV}"
S = "${WORKDIR}/git"

# DEPENDS para compilación - SOLO lo esencial
DEPENDS = " \
    python3-native \
"

# RDEPENDS para runtime
RDEPENDS:${PN} = " \
    python3-opencv \
    python3-numpy \
    tensorflow-lite \
    python3-ctypes \
    python3-json \
    bash \
    gstreamer1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
"

do_install() {
    install -d ${D}${bindir}
    install -d ${D}${datadir}/semaforo
    
    # Crear wrapper script ejecutable
    cat > ${D}${bindir}/semaforo << 'SCRIPT'
#!/bin/bash
echo "========================================"
echo "    SEMÁFORO INTELIGENTE - YOLO TFLite"
echo "========================================"

# Configurar display (X11 debería estar corriendo en segundo plano)
export DISPLAY=:0
export XAUTHORITY=/home/root/.Xauthority

# Verificar si X11 está corriendo
if ! xset q >/dev/null 2>&1; then
    echo "Iniciando X11 automáticamente..."
    startx > /dev/null 2>&1 &
    sleep 3
    echo " X11 iniciado"
else
    echo "X11 ya está corriendo"
fi

# Navegar al directorio de la aplicación
cd /usr/share/semaforo

# Verificar archivos esenciales
echo "Verificando archivos..."
[ -f "semaforo.py" ] && echo "T semaforo.py" || echo "X semaforo.py - NO ENCONTRADO"
[ -f "video_test.mp4" ] && echo "T video_test.mp4" || echo "X video_test.mp4 - NO ENCONTRADO"
[ -f "yolo11n_float16.tflite" ] && echo "T yolo11n_float16.tflite" || echo "X modelo - NO ENCONTRADO"
[ -f "labels.txt" ] && echo "T labels.txt" || echo "X labels.txt - NO ENCONTRADO"

echo ""
echo "Ejecutando aplicación..."
echo "Presiona 'q' en la ventana para salir"
echo "========================================"

# Ejecutar la aplicación Python
python3 semaforo.py

echo ""
echo "Aplicación terminada."
SCRIPT
    chmod 0755 ${D}${bindir}/semaforo
    
    # Instalar archivos de la aplicación
    install -m 0755 ${S}/src/semaforo.py ${D}${datadir}/semaforo/
    install -m 0644 ${S}/src/yolo11n_float16.tflite ${D}${datadir}/semaforo/
    install -m 0644 ${S}/src/labels.txt ${D}${datadir}/semaforo/
    install -m 0644 ${S}/src/video_test.mp4 ${D}${datadir}/semaforo/
}

FILES:${PN} += " \
    ${bindir}/semaforo \
    ${datadir}/semaforo/semaforo.py \
    ${datadir}/semaforo/yolo11n_float16.tflite \
    ${datadir}/semaforo/labels.txt \
    ${datadir}/semaforo/video_test.mp4 \
"

INSANE_SKIP:${PN} += "already-stripped"
