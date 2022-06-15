#!/usr/bin/env bash

set -ue

USER_ID=${USER_ID:-$(id -u)}

USER=${DOCKER_USER_NAME:-runner}

WORK_TREE_DIR=${PROJECT_HOME_DIR:-"/home"}

HOME_DIR="/home/${USER}"

mkdir -p "${WORK_TREE_DIR}"

if ! id -u "${USER_ID}" &> /dev/null
then
    echo "Create a new user with ID: ${USER_ID}"
    mkdir -p "$HOME_DIR"

    useradd -d "$HOME_DIR" -u ${USER_ID} ${USER}

    chown -R ${USER} "$HOME_DIR"
    chown -R ${USER} "${WORK_TREE_DIR}"
else
    USER=$(id -un $USER_ID)
    echo "No user specified! user=${USER} with id=${USER_ID} is selected."
    echo "Specify user via environment variable!"
fi

cd "${WORK_TREE_DIR}"

gosu ${USER} "$@"