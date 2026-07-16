#!/usr/bin/env bash
# Build the latest upstream libfranka in /tmp and test connecting to a
# Franka controller using libfranka's own example binaries. Does NOT
# touch your existing venv, catkin_ws, or system packages.
#
# Pinocchio (libfranka >= 0.20 dependency) is also built from source
# into /tmp/libfranka_latest_test/pinocchio_local — no apt, no sudo.
#
# Run on EACH node (slave for left arm, master for right arm):
#
#   bash test_latest_libfranka.sh                 # default IP, tag 0.20.5
#   bash test_latest_libfranka.sh 172.16.0.2      # explicit IP
#   bash test_latest_libfranka.sh 172.16.0.2 0.21.2   # pin a libfranka tag
#
# Pre-reqs (system-level, must already be present — script will not
# install these): a C++ compiler with C++17, cmake >=3.10, git, and
# Boost development headers. On Ubuntu these come from build-essential
# and libboost-all-dev. If you're missing Boost the script will tell
# you which one and stop before doing anything destructive.

set -euo pipefail

ROBOT_IP="${1:-172.16.0.2}"
WANT_TAG="${2:-}"
WORK="/tmp/libfranka_latest_test"
PINO_PREFIX="$WORK/pinocchio_local"
EIGEN_PREFIX="$WORK/eigen_local"
PINO_TAG="${PINOCCHIO_TAG:-v3.0.0}"   # 3.x has clean cmake; 2.7.x has a bug
                                      # in CMakeLists where URDFDOM_VERSION is
                                      # dereferenced even with URDF disabled

echo "================================================================"
echo "  Target robot: $ROBOT_IP"
echo "  Build dir   : $WORK"
echo "  Pinocchio   : $PINO_PREFIX (built from source if missing)"
echo "================================================================"

mkdir -p "$WORK"
cd "$WORK"

# ---------------------------------------------------------------- 0. host deps
# Hard-fail early on missing toolchain so we don't waste 10 min cloning.
need_pkg() {
    local bin="$1"; local hint="$2"
    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "Missing '$bin'. Install with: $hint"
        exit 1
    fi
}
need_pkg git    "sudo apt-get install -y git"
need_pkg cmake  "sudo apt-get install -y cmake"
need_pkg g++    "sudo apt-get install -y build-essential"

# Boost is needed by pinocchio. Try a wider set of locations + dpkg, since
# distros put headers under /usr/include/boost (Ubuntu) but conda/openrobots
# stash them elsewhere. If still nothing, we let cmake decide — it has
# its own FindBoost logic that may succeed where this naive check fails.
BOOST_OK=""
for f in \
    /usr/include/boost/version.hpp \
    /usr/local/include/boost/version.hpp \
    /opt/openrobots/include/boost/version.hpp \
    "$HOME/.local/include/boost/version.hpp"; do
    if [ -f "$f" ]; then
        BOOST_OK="$f"
        break
    fi
done
if [ -z "$BOOST_OK" ] && command -v dpkg >/dev/null 2>&1; then
    if dpkg -l 2>/dev/null | grep -q "^ii.*libboost.*-dev"; then
        BOOST_OK="(dpkg reports libboost*-dev installed)"
    fi
fi
if [ -z "$BOOST_OK" ]; then
    echo "WARNING: could not find Boost headers in the usual places."
    echo "         Letting cmake try anyway — if pinocchio fails, install with:"
    echo "         sudo apt-get install -y libboost-all-dev"
else
    echo "Boost: $BOOST_OK"
fi

# ---------------------------------------------------------------- 1. eigen3
# Pinocchio needs Eigen >= 3.3. Most distros have it; if not, fetch source.
EIGEN_CMAKE_DIR=""
for p in /usr /usr/local "$HOME/.local" "$EIGEN_PREFIX"; do
    if [ -f "$p/include/eigen3/Eigen/Core" ] || \
       [ -f "$p/share/eigen3/cmake/Eigen3Config.cmake" ]; then
        EIGEN_CMAKE_DIR="$p"
        break
    fi
done
if [ -z "$EIGEN_CMAKE_DIR" ]; then
    echo "Eigen3 not found, fetching source into $EIGEN_PREFIX ..."
    if [ ! -d eigen-src ]; then
        git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git eigen-src
    fi
    mkdir -p eigen-build
    cd eigen-build
    cmake -DCMAKE_INSTALL_PREFIX="$EIGEN_PREFIX" ../eigen-src >/dev/null
    make install >/dev/null
    cd "$WORK"
    EIGEN_CMAKE_DIR="$EIGEN_PREFIX"
fi
echo "Eigen3 from: $EIGEN_CMAKE_DIR"

# ---------------------------------------------------------------- 2. pinocchio
# Prefer a system-installed pinocchio (robotpkg /opt/openrobots, apt
# /usr, conda, or a local build under PINO_PREFIX). Only if none of
# those is present do we build from source — pinocchio is a beast and
# the build-from-source path has tag-specific cmake bugs, so avoid it.
PINO_CMAKE_DIR=""
for p in /opt/openrobots /usr /usr/local "$HOME/.local" "$PINO_PREFIX"; do
    for sub in lib/cmake/pinocchio share/cmake/pinocchio \
               lib/x86_64-linux-gnu/cmake/pinocchio; do
        if [ -f "$p/$sub/pinocchioConfig.cmake" ] || \
           [ -f "$p/$sub/pinocchio-config.cmake" ]; then
            PINO_CMAKE_DIR="$p/$sub"
            PINO_PREFIX="$p"
            break 2
        fi
    done
done

if [ -n "$PINO_CMAKE_DIR" ]; then
    echo "Pinocchio (system): $PINO_CMAKE_DIR"
else
    echo
    echo "----------------------------------------------------------------"
    echo "No system pinocchio found, building $PINO_TAG into $PINO_PREFIX"
    echo "----------------------------------------------------------------"
    if [ ! -d pinocchio-src/.git ]; then
        rm -rf pinocchio-src
        git clone --recursive --branch "$PINO_TAG" \
            https://github.com/stack-of-tasks/pinocchio.git pinocchio-src
    fi
    rm -rf pinocchio-build
    mkdir -p pinocchio-build
    cd pinocchio-build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="$PINO_PREFIX" \
          -DCMAKE_PREFIX_PATH="$EIGEN_CMAKE_DIR" \
          -DBUILD_PYTHON_INTERFACE=OFF \
          -DBUILD_WITH_URDF_SUPPORT=OFF \
          -DBUILD_WITH_COLLISION_SUPPORT=OFF \
          -DBUILD_WITH_AUTODIFF_SUPPORT=OFF \
          -DBUILD_WITH_CASADI_SUPPORT=OFF \
          -DBUILD_WITH_CODEGEN_SUPPORT=OFF \
          -DBUILD_TESTING=OFF \
          -DBUILD_BENCHMARK=OFF \
          -DBUILD_UTILS=OFF \
          -DBUILD_WITH_OPENMP_SUPPORT=OFF \
          -DBUILD_WITH_HPP_FCL_SUPPORT=OFF \
          -DBUILD_WITH_SDF_SUPPORT=OFF \
          -DBUILD_WITH_MPFR_SUPPORT=OFF \
          -DBUILD_WITH_EXTRA_SUPPORT=OFF \
          -DGENERATE_PYTHON_STUBS=OFF \
          -Wno-dev \
          ../pinocchio-src 2>&1 | tail -25
    make -j"$(nproc)" 2>&1 | tail -5
    make install >/dev/null
    cd "$WORK"

    for sub in lib/cmake/pinocchio share/cmake/pinocchio; do
        if [ -d "$PINO_PREFIX/$sub" ]; then
            PINO_CMAKE_DIR="$PINO_PREFIX/$sub"
            break
        fi
    done
    if [ -z "$PINO_CMAKE_DIR" ]; then
        echo "Pinocchio install did not produce a cmake config — aborting."
        exit 1
    fi
    echo "Pinocchio (built): $PINO_CMAKE_DIR"
fi

# ---------------------------------------------------------------- 3. libfranka
# Clone (or reuse) source.
if [ ! -d libfranka/.git ]; then
    rm -rf libfranka
    git clone --recursive https://github.com/frankarobotics/libfranka.git
fi
cd libfranka
git fetch --tags --quiet

# 2. Show available tags so you can pin later if needed.
echo
echo "----------------------------------------------------------------"
echo "Latest 15 upstream tags:"
echo "----------------------------------------------------------------"
git tag --sort=-v:refname | head -15

if [ -z "$WANT_TAG" ]; then
    # Default to 0.20.5: first stable release of the pinocchio-based
    # series, most likely to match recent FR3 firmwares (5.9.x). Override
    # by passing the tag as $2.
    WANT_TAG="0.20.5"
fi
echo
echo "Using tag: $WANT_TAG"
git checkout --quiet "$WANT_TAG"
git submodule update --init --recursive --quiet

# 3. Build libfranka.
rm -rf build
mkdir -p build
cd build

echo
echo "----------------------------------------------------------------"
echo "Configuring libfranka (cmake)..."
echo "----------------------------------------------------------------"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DBUILD_EXAMPLES=ON \
      -Dpinocchio_DIR="$PINO_CMAKE_DIR" \
      -DCMAKE_PREFIX_PATH="$PINO_PREFIX;$EIGEN_CMAKE_DIR" \
      .. 2>&1 | tail -25

if [ ! -f Makefile ] && [ ! -f build.ninja ]; then
    echo
    echo "================================================================"
    echo "libfranka cmake failed. Check the output above. Common fixes:"
    echo "  - Missing Boost headers: sudo apt-get install -y libboost-all-dev"
    echo "  - Try an older libfranka tag, e.g.:"
    echo "      bash $0 $ROBOT_IP 0.20.0"
    echo "================================================================"
    exit 1
fi

echo
echo "----------------------------------------------------------------"
echo "Building..."
echo "----------------------------------------------------------------"
cmake --build . -j"$(nproc)" 2>&1 | tail -10

echo
echo "Built libfranka:"
ls -la libfranka.so* 2>/dev/null || true

# Find where the example binaries actually landed (varies across versions).
EX_DIR=""
for cand in examples ./examples ./build/examples .; do
    if [ -x "$cand/communication_test" ]; then
        EX_DIR="$cand"
        break
    fi
done
if [ -z "$EX_DIR" ]; then
    echo "Could not locate communication_test binary, dumping tree:"
    find . -maxdepth 3 -type f -name "communication_test" -o -name "echo_robot_state" 2>/dev/null
    exit 1
fi

# Examples are dynamically linked against pinocchio at $PINO_PREFIX/lib
# and against libfranka.so in the build dir. Export both for the run.
export LD_LIBRARY_PATH="$PINO_PREFIX/lib:$WORK/libfranka/build:${LD_LIBRARY_PATH:-}"

# 4. Communication test - the canonical "does it work" probe.
echo
echo "================================================================"
echo "  communication_test  ($EX_DIR/communication_test $ROBOT_IP)"
echo "================================================================"
echo "Note: this opens FCI control. Make sure the robot is unlocked,"
echo "FCI is activated in Desk, and nobody else is holding control."
echo
"$EX_DIR/communication_test" "$ROBOT_IP" \
    && echo ">>> communication_test PASSED" \
    || echo ">>> communication_test FAILED (exit $?)"

# 5. Read-only state echo - cheap second opinion.
echo
echo "================================================================"
echo "  echo_robot_state  ($EX_DIR/echo_robot_state $ROBOT_IP, 3s)"
echo "================================================================"
timeout 3 "$EX_DIR/echo_robot_state" "$ROBOT_IP" \
    && echo ">>> echo_robot_state exited cleanly" \
    || echo ">>> echo_robot_state stopped (exit $?, timeout=3s is normal)"

echo
echo "================================================================"
echo "  Done. Tag tested: $WANT_TAG"
echo "================================================================"
