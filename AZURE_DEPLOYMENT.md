# Azure Deployment Setup Guide

This guide explains how to set up GitHub Actions to automatically deploy scripts to Azure Blob Storage for use with Azure Batch via Azure Data Factory.

## Prerequisites

1. An Azure subscription
2. An Azure Storage Account
3. An Azure Service Principal with appropriate permissions

## Azure Setup

### 1. Create Azure Storage Account

```bash
# Create a resource group (if needed)
az group create --name customer-insights-rg --location eastus

# Create a storage account
az storage account create \
  --name customerinsightsstorage \
  --resource-group customer-insights-rg \
  --location eastus \
  --sku Standard_LRS

# Create a container for scripts
az storage container create \
  --name scripts \
  --account-name customerinsightsstorage
```

### 2. Create Azure Service Principal

Create a service principal with contributor access to the storage account:

```bash
# Create service principal
az ad sp create-for-rbac \
  --name "github-actions-customer-insights" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/customer-insights-rg \
  --sdk-auth

# Grant Storage Blob Data Contributor role
az role assignment create \
  --assignee {service-principal-app-id} \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/{subscription-id}/resourceGroups/customer-insights-rg/providers/Microsoft.Storage/storageAccounts/customerinsightsstorage
```

This will output JSON credentials that you'll use in the next step.

## GitHub Secrets Configuration

Add the following secrets to your GitHub repository (`Settings` → `Secrets and variables` → `Actions`):

### Required Secrets

1. **AZURE_CREDENTIALS**
   - Value: The entire JSON output from the service principal creation command
   - Format:
     ```json
     {
       "clientId": "...",
       "clientSecret": "...",
       "subscriptionId": "...",
       "tenantId": "...",
       "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
       "resourceManagerEndpointUrl": "https://management.azure.com/",
       "activeDirectoryGraphResourceId": "https://graph.windows.net/",
       "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
       "galleryEndpointUrl": "https://gallery.azure.com/",
       "managementEndpointUrl": "https://management.core.windows.net/"
     }
     ```

2. **AZURE_STORAGE_ACCOUNT_NAME**
   - Value: Your storage account name (e.g., `customerinsightsstorage`)

## Workflow Triggers

The GitHub Actions workflow will automatically run when:

1. **Push to main branch** - When changes are made to:
   - `scripts/**` directory
   - `src/**` directory

2. **Manual trigger** - You can manually trigger the workflow from the Actions tab in GitHub

## What Gets Deployed

The workflow performs the following steps:

1. **Validation** - Lints all Python scripts for syntax errors
2. **Container Setup** - Ensures the `scripts` container exists in Azure Storage
3. **Upload Scripts** - Uploads individual Python files from `scripts/` and `src/` directories
4. **Script Inventory** - Creates a JSON inventory file with versioning information
5. **Verification** - Lists all uploaded files

Files uploaded to Azure Blob Storage:
- All Python scripts from the `scripts/` directory → `scripts/` folder in blob storage
- All Python source code from the `src/` directory → `src/` folder in blob storage
- The `requirements.txt` file → root of blob storage
- Script inventory JSON → `inventory/script_inventory.json`

## Using with Azure Data Factory

Once scripts are uploaded to Azure Blob Storage, you can configure Azure Data Factory to:

1. Create a pipeline that triggers Azure Batch jobs
2. Mount the blob container to Azure Batch compute nodes
3. Execute the scripts (e.g., `run_daily_pipeline.py`, `run_weekly_pipeline.py`)

### Example Azure Data Factory Pipeline

Create a Custom Activity in Azure Data Factory that:

1. References your Azure Batch account
2. Mounts the blob storage container with scripts
3. Executes commands like:
   ```bash
   python scripts/run_daily_pipeline.py
   ```

## Verifying Deployment

After the workflow runs, you can verify the deployment:

```bash
# List uploaded files
az storage blob list \
  --account-name customerinsightsstorage \
  --container-name scripts \
  --auth-mode login \
  --output table

# Download a file to verify
az storage blob download \
  --account-name customerinsightsstorage \
  --container-name scripts \
  --name scripts/run_daily_pipeline.py \
  --file downloaded_script.py \
  --auth-mode login
```

## Troubleshooting

### Authentication Issues

If you see authentication errors:
1. Verify the AZURE_CREDENTIALS secret is valid JSON
2. Ensure the service principal has the correct roles
3. Check that the service principal hasn't expired

### Upload Failures

If script uploads fail:
1. Verify the storage account name is correct
2. Check that the container exists
3. Ensure the service principal has "Storage Blob Data Contributor" role

### Workflow Not Triggering

If the workflow doesn't run automatically:
1. Ensure changes are pushed to the `main` branch
2. Verify the modified files are in `scripts/` or `src/` directories
3. Check the Actions tab for any errors

## Security Best Practices

1. **Never commit Azure credentials** to the repository
2. **Use managed identities** when possible instead of service principals
3. **Rotate credentials regularly** - Update the service principal secrets periodically
4. **Limit access scope** - Only grant minimum required permissions
5. **Use separate storage accounts** for different environments (dev, staging, prod)

## Additional Resources

- [Azure CLI Storage Commands](https://docs.microsoft.com/en-us/cli/azure/storage)
- [Azure Batch Documentation](https://docs.microsoft.com/en-us/azure/batch/)
- [Azure Data Factory Custom Activities](https://docs.microsoft.com/en-us/azure/data-factory/transform-data-using-dotnet-custom-activity)
- [GitHub Actions Azure Login](https://github.com/marketplace/actions/azure-login)
