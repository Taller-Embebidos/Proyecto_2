Guía de compilación Yocto para Raspberry Pi 4

Entorno: RHEL 10 / Fedora 42 usando Toolbx o Podman <br>
1. Preparación del entorno

Instalación de Toolbox en un host Fedora / RHEL 10
```bash
sudo dnf install toolbox
```
Si desea usar Podman

```bash
sudo dnf -y install podman
```

Crear el directorio de trabajo:
```bash
mkdir ~/tools
```
Creación del contenedor

Usando Toolbox (recomendado):
```bash
toolbox create --image registry.fedoraproject.org/fedora-toolbox:40 yocto
```

Usando Podman:
```bash
podman run -it --name yocto -v /home/<user>/tools:/tools:z registry.fedoraproject.org/fedora-toolbox:40 /bin/bash
```

Notas:

    En este caso se comparte la carpeta `~/tools` del host con el contenedor <br>
    Cambiar <user> por el usuario del equipo host <br>

    Se utiliza la imagen de Fedora 40 como contenedor con un alias 'yocto'

2. Ingreso al contenedor

Usando Toolbox:
```bash
toolbox enter yocto
```

Usando Podman:
```bash
podman start -ai yocto
```

Consideraciones:

    En Podman se requiere un usuario sin privilegios de root (Yocto no permite compilar como root)<br>

    En Toolbox esto no es necesario, ya que las carpetas del host están expuestas y se ejecuta con el usuario del sistema

Crear usuario no root en podman:
```bash
useradd -m -u 1000 -s /bin/bash build
```
El usuario se llamará build.

3. Instalación de dependencias

Ingresar en usuario root solo para instalar los paquetes (aplica para Toolbox, en el caso de podman ya estamos en usuario root)

```bash
su
```

Instalar los paquetes requeridos para la compilación:
```bash
dnf install -y @development-tools bzip2 ccache chrpath cpio cpp diffstat diffutils file findutils gawk gcc gcc-c++ git glibc-devel glibc-langpack-en gzip hostname lz4 make patch perl perl-Data-Dumper perl-File-Compare perl-File-Copy perl-FindBin perl-Text-ParseWords perl-Thread-Queue perl-bignum perl-locale python python3 python3-devel python3-GitPython python3-jinja2 python3-pexpect python3-pip python3-setuptools rpcgen socat tar texinfo unzip wget which xz zstd SDL-devel xterm mesa-libGL-devel nano sudo
```

4. Configuración de idioma

```bash
dnf install -y glibc-all-langpacks
echo 'LANG=en_US.UTF-8' > /etc/locale.conf
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```
Finalmente ya puede salir de superusuario

```bash
exit
```
Y volverá a usuario normal

En podman, requerimos entrar en modo usuario para compilar

```bash
su - build
```

Salir y volver a root:
```bash
exit
```

5. Preparación del entorno Yocto

```bash
cd ~/tools
git clone git://git.yoctoproject.org/poky
cd poky
git checkout -t origin/kirkstone -b my-kirkstone
git pull
```
Nota:
Usando Podman, la carpeta esta en /tools, en el caso de toolbx la carpeta creada esta en /home/<user>/tools. <br>

Inicializar el entorno:
```bash
source oe-init-build-env rpi-build
```
6. Agregar capa de Raspberry Pi

```bash
cd ~/tools/poky
git clone https://git.yoctoproject.org/meta-raspberrypi
cd meta-raspberrypi/
git checkout -t origin/kirkstone -b my-kirkstone
git pull
```

Registrar la capa en bblayers.conf:
```bash
cd ~/tools/poky/rpi-build
bitbake-layers add-layer ../meta-raspberrypi
```

7. Configuración de compilación para Raspberry Pi 4

Editar conf/local.conf:
```bash
cd ~/tools/poky/rpi-build/
nano conf/local.conf
```

Modificar dentro de local.conf:
```bash
MACHINE ??= "raspberrypi4"
EXTRA_IMAGE_FEATURES ?= "debug-tweaks tools-sdk tools-debug"
BB_HASHSERVE_UPSTREAM = "hashserv.yoctoproject.org:8686"
SSTATE_MIRRORS ?= "file://.* http://sstate.yoctoproject.org/all/PATH;downloadfilename=PATH"
```


Nota: Aquí puede incluir su capa de personalización "custom" si aplica. Comenta la línea por defecto de 'MACHINE' para que la compilación sea de la Raspberry pi4

8. Descarga de dependencias para imagen mínima (opcional), para compilar offline

```
bitbake core-image-minimal -c fetch


```
9. Compilación de imagen mínima 

```
bitbake core-image-minimal
```

10. Generación y copia de imagen en la SD de la Raspberrypi

```bash
cd ~/tools/poky/rpi-build/tmp/deploy/images/
sudo bmaptool copy core-image-minimal.rootfs.wic.bz2 --bmap core-image-minimal.rootfs.wic.bmap <device>
```

Referencias

[Site] https://docs.yoctoproject.org/ref-manual/system-requirements.html

[Site] https://git.yoctoproject.org/meta-raspberrypi/

[Site] https://docs.yoctoproject.org/brief-yoctoprojectqs/

[Site] https://github.com/agherzan/meta-raspberrypi
