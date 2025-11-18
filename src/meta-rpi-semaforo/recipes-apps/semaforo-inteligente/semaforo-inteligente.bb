SUMMARY = "Sistema de semáforo inteligente con YOLO"
LICENSE = "CLOSED"

SRC_URI = "git://github.com/Taller-Embebidos/Proyecto_2.git;branch=main;protocol=https"
SRCREV = "${AUTOREV}"
S = "${WORKDIR}/git"

RDEPENDS:${PN} = " \
    python3-opencv \
    python3-numpy \
    tflite-runtime \
"

do_install() {
    install -d ${D}${bindir}
    install -m 0755 ${S}/src/semaforo.py ${D}${bindir}/semaforo
    install -m 0644 ${S}/src/yolo11n_float16.tflite ${D}${bindir}/
    install -m 0644 ${S}/src/labels.txt ${D}${bindir}/
    install -m 0644 ${S}/src/video_test.mp4 ${D}${bindir}/
}

FILES:${PN} += " \
    ${bindir}/semaforo \
    ${bindir}/yolo11n_float16.tflite \
    ${bindir}/labels.txt \
    ${bindir}/video_test.mp4 \
"
