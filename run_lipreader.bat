@echo off
rem Launch LipReader live in the conda env 'lipreader'
call C:\Users\susha\anaconda3\Scripts\activate.bat lipreader
python live_lipreader.py %*
