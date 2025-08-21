#!/bin/bash

AZURE_APP_ID=$1
AZURE_CLIENT_SECRET=$2
AZURE_TENANT_ID=$3
ACR_REGISTRY=$4
REPOSITORY_1=$5
TAG=$6
echo $ACR_REGISTRY
[[ -z "$ACR_REGISTRY" ]] && exit 1  # registry name must be passed as parameter

if [[ -z $AZURE_APP_ID || -z $AZURE_CLIENT_SECRET  || -z $AZURE_TENANT_ID ]]
then
    echo 'The Databricks host URL and secret access token must be passed from job VM'
    exit 1
fi
echo $HOME
az login --service-principal --username $AZURE_APP_ID --password $AZURE_CLIENT_SECRET --tenant $AZURE_TENANT_ID
az account set --subscription "eb6f96f0-ff9e-450e-b2a6-5519de91feb9"
az account show --output table
az acr show --name $ACR_REGISTRY
az acr login --name $ACR_REGISTRY 
docker image ls
docker tag $REPOSITORY_1:$TAG $ACR_REGISTRY.azurecr.io/$REPOSITORY_1:$TAG
docker push $ACR_REGISTRY.azurecr.io/$REPOSITORY_1:$TAG