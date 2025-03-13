#!/bin/bash
set -e

# Hardcoded Configuration
BACALHAU_NODE_DIR="/bacalhau_node"
BACALHAU_DATA_DIR="/bacalhau_data"
CONTAINER_NAME="bacalhau_node-bacalhau-node"

# -------------------------------------------------------------------
# Lightweight IP Address Detection
# -------------------------------------------------------------------

# Get Public IP (using a lightweight external service)
get_public_ip() {
    curl -s --max-time 2 https://ip.me || echo "UNKNOWN"
}

# Get Private IP (using hostname -I)
get_private_ip() {
    hostname -I | awk '{print $1}' || echo "UNKNOWN"
}

# -------------------------------------------------------------------
# Write Node Info
# -------------------------------------------------------------------

# Ensure the node directory exists
mkdir -p "${BACALHAU_NODE_DIR}"

# Write node-info file
cat > "${BACALHAU_NODE_DIR}/node-info" << EOF
PUBLIC_IP=$(get_public_ip)
PRIVATE_IP=$(get_private_ip)
EOF

# -------------------------------------------------------------------
# Verify Docker Service
# -------------------------------------------------------------------

echo "Verifying Docker service..."
if ! systemctl is-active --quiet docker; then
    echo "Docker is not running. Starting Docker..."
    systemctl start docker
    sleep 5  # Give Docker time to start
fi

# -------------------------------------------------------------------
# Validate Configuration
# -------------------------------------------------------------------

echo "Setting up configuration..."
if [ -f "${BACALHAU_NODE_DIR}/config.yaml" ]; then
    echo "Configuration file exists at ${BACALHAU_NODE_DIR}/config.yaml"
else
    echo "Error: Configuration file not found at ${BACALHAU_NODE_DIR}/config.yaml"
    exit 1
fi

# -------------------------------------------------------------------
# Start Docker Compose Services
# -------------------------------------------------------------------

echo "Starting Docker Compose services..."
if [ -f "${BACALHAU_NODE_DIR}/docker-compose.yaml" ]; then
    cd "${BACALHAU_NODE_DIR}" || exit
    echo "Stopping and removing any existing containers..."
    docker compose down
    if docker ps -a | grep -q "${CONTAINER_NAME}"; then
        echo "Found stray containers, removing them..."
        docker ps -a | grep "${CONTAINER_NAME}" | awk '{print $1}' | xargs -r docker rm -f
    fi
    echo "Pulling latest images..."
    docker compose pull
    echo "Starting services..."
    docker compose up -d
    echo "Docker Compose started."
else
    echo "Error: docker-compose.yaml not found at ${BACALHAU_NODE_DIR}/docker-compose.yaml"
    exit 1
fi

# -------------------------------------------------------------------
# Final Output
# -------------------------------------------------------------------

echo "Bacalhau node setup complete."
echo "Public IP: $(get_public_ip)"
echo "Private IP: $(get_private_ip)"

exit 0