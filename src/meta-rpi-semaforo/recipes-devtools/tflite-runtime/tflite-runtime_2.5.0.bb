SUMMARY = "TensorFlow Lite Runtime (precompiled)"
LICENSE = "Apache-2.0"
LIC_FILES_CHKSUM = "file://${WORKDIR}/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl;md5=551c6917f1ba67154076b26673520691"

SRC_URI = " \
    https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl;name=whl \
"

SRC_URI[whl.sha256sum] = "9175f1bb1c2f1f5c921117735a81943f85411248d78781453435a9bbfc212b91"

S = "${WORKDIR}"

DEPENDS = "python3-native unzip-native"
inherit python3native python3-dir

do_install() {
    # Crear directorios de destino
    install -d ${D}${PYTHON_SITEPACKAGES_DIR}/tflite_runtime
    install -d ${D}${PYTHON_SITEPACKAGES_DIR}/tflite_runtime-2.5.0.post1.dist-info
    
    # Extraer el wheel a un directorio temporal
    cd ${WORKDIR}
    unzip -q -d temp_extract tflite_runtime-2.5.0.post1-cp37-cp37m-linux_armv7l.whl
    
    # Copiar archivos del paquete tflite_runtime
    if [ -d "temp_extract/tflite_runtime-2.5.0.post1.data/purelib/tflite_runtime" ]; then
        cp -r temp_extract/tflite_runtime-2.5.0.post1.data/purelib/tflite_runtime/* ${D}${PYTHON_SITEPACKAGES_DIR}/tflite_runtime/
    fi
    
    # Copiar metadata del paquete
    if [ -d "temp_extract/tflite_runtime-2.5.0.post1.dist-info" ]; then
        cp -r temp_extract/tflite_runtime-2.5.0.post1.dist-info/* ${D}${PYTHON_SITEPACKAGES_DIR}/tflite_runtime-2.5.0.post1.dist-info/
    fi
    
    # Limpiar
    rm -rf temp_extract
    
    # Corregir permisos de manera segura
    if [ -f "${D}${PYTHON_SITEPACKAGES_DIR}/tflite_runtime/_pywrap_tensorflow_interpreter_wrapper.cpython-37m-arm-linux-gnueabihf.so" ]; then
        chmod 644 "${D}${PYTHON_SITEPACKAGES_DIR}/tflite_runtime/_pywrap_tensorflow_interpreter_wrapper.cpython-37m-arm-linux-gnueabihf.so"
    fi
}

FILES:${PN} += "\
    ${PYTHON_SITEPACKAGES_DIR}/tflite_runtime/ \
    ${PYTHON_SITEPACKAGES_DIR}/tflite_runtime/* \
    ${PYTHON_SITEPACKAGES_DIR}/tflite_runtime-2.5.0.post1.dist-info/ \
    ${PYTHON_SITEPACKAGES_DIR}/tflite_runtime-2.5.0.post1.dist-info/* \
"

INSANE_SKIP:${PN} += "already-stripped file-rdeps"
RDEPENDS:${PN} = "python3-core python3-numpy"
