# Forzar Python 3.12 explícitamente
export PYTHON_EXECUTABLE = "${bindir}/python3.12"
export PYTHON_LIBRARY = "${libdir}/libpython3.12.so"
export PYTHON_INCLUDE_DIR = "${includedir}/python3.12"

do_configure:prepend() {
    export PYTHON=python3.12
    export PYTHON_CONFIG=python3.12-config
}

# Verificar que use Python 3.12
do_configure:append() {
    echo "=== Verificando Python ==="
    ${PYTHON} --version
    echo "PYTHON: ${PYTHON}"
    echo "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}"
}
