https://www.kaggle.com/c/understanding_cloud_organization

# Prepare the data

```sh
# Make sure you sign in and have agree to join the competetion and get your kaggle.json ready.
# from this from root directory
kaggle competitions download -c understanding_cloud_organization
```

# Unzip the data

```
mkdir input notebook
mv *.csv input
mv *.zip input
cd input
unzip train.csv.zip
mkdir train test
unzip -d train train_images.zip
unzip -d test test_images.zip
```

You should have a folder structure like this
input
|**train
|**test
notebook
|\_\_notebook1.ipynb
