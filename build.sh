echo "Are you sure you want to remove the build and dist directories? [y/N]"
read confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    rm -rf ./build
    rm -rf ./dist
else
    echo "Aborted."
    exit 1
fi

python -m build

echo -e "\nTo upload to PyPI, run the following command:\n"
echo -e "\ttwine upload dist/*\n"
echo -e "When prompted for credentials, use the following:\n"
echo -e "\tUsername: __token__"
echo -e "\tPassword: <Your PyPI API token>\n"
