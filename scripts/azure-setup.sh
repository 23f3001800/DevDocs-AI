#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Azure App Service Setup for DevDocs-AI
# Run this ONCE to create all Azure resources.
# Prerequisites: az CLI installed + logged in (az login)
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ────────────────────────────────────────────
RESOURCE_GROUP="devdocs-ai-rg"
LOCATION="centralindia"          # closest to you — change if needed
APP_SERVICE_PLAN="devdocs-ai-plan"
APP_NAME="devdocs-ai"            # must be globally unique
SKU="B1"                         # 1.75GB RAM, ~$13/month

echo "🔧 Creating resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "🔧 Creating App Service plan (B1 — 1.75GB RAM, Linux)..."
az appservice plan create \
  --name "$APP_SERVICE_PLAN" \
  --resource-group "$RESOURCE_GROUP" \
  --sku "$SKU" \
  --is-linux

echo "🔧 Creating Web App with Docker container..."
az webapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --deployment-container-image-name "ghcr.io/23f3001800/devdocs-ai:latest"

echo "🔧 Enabling GHCR access (public image)..."
az webapp config container set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --container-image-name "ghcr.io/23f3001800/devdocs-ai:latest" \
  --container-registry-url "https://ghcr.io"

echo "🔧 Setting environment variables..."
az webapp config appsettings set \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings \
    WEBSITES_PORT=8000 \
    PYTHONUNBUFFERED=1 \
    LANGSMITH_TRACING=true

echo ""
echo "⚠️  Set these secrets manually in the Azure portal or CLI:"
echo "   az webapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP --settings \\"
echo '     ANTHROPIC_API_KEY=<your-key> \'
echo '     GOOGLE_API_KEY=<your-key> \'
echo '     JWT_SECRET=<random-32-char-string> \'
echo '     ADMIN_PASSWORD=<your-admin-password>'
echo ""

echo "🔧 Creating service principal for GitHub Actions..."
SP_JSON=$(az ad sp create-for-rbac \
  --name "devdocs-ai-github" \
  --role contributor \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP" \
  --sdk-auth)

echo ""
echo "✅ Done! Your app will be live at: https://$APP_NAME.azurewebsites.net"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Copy this JSON and add it as a GitHub secret:"
echo "   Name: AZURE_CREDENTIALS"
echo "   Value:"
echo "$SP_JSON"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
