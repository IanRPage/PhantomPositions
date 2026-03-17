@echo off

set "URL=https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction"
set "ZIP_PATH=data\real-or-fake-fake-jobposting-prediction.zip"
set "DEST=data"

curl -L -o "%ZIP_PATH%" "%URL%"

tar -xf "%ZIP_PATH%" -C "%DEST%"

del "%ZIP_PATH%"
