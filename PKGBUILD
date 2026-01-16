pkgname=vkdt-denox
pkgver=0.0.1
pkgrel=1
pkgdesc="denox model codegeneration for vkdt"
arch=('x86_64')
url="https://github.com/kistenklaus/vkdt-denox"
license=('MIT')
depends=()
optdepends=(
  'fmt: use system fmt instead of bundled'
)

makedepends=(
  'cmake'
  'ninja'
  'flatbuffers'
)

source=("vkdt-denox-$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
  cd "$srcdir/vkdt-denox-$pkgver"

  cmake -S . -B build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DCMAKE_UNITY_BUILD=ON \
    -DVKDT_DENOX_STRICT_WARNINGS=OFF \
    -DVKDT_DENOX_SAN=OFF \
    -DVKDT_DENOX_USE_SYSTEM_FLATBUFFERS=ON 

  cmake --build build
}

package() {
  cd "$srcdir/vkdt-denox-$pkgver"

  DESTDIR="$pkgdir" cmake --install build
  install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
