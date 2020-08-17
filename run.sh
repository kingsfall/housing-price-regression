
echo "run.sh will first install dependent libraries from requirements.txt"
! pip3 install -r requirements.txt
echo "successfully installed requirements"
OPTION=0
while [ $OPTION != 3 ]
do
    echo "please type either 1, 2, 3: "
    echo "1. run module1.py"
    echo "2. run module2.py"
    echo "3. exit run.sh script"

    read OPTION
    if [ $OPTION == 1 ]
    then
        ! python3 src/module1.py
    elif [ $OPTION == 2 ]
    then
        ! python3 src/module2.py
    elif [ $OPTION == 3 ]
    then
        echo "exiting script"
    else
        echo "option not recognized. please try again"
    fi
done