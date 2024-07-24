# create "logs" folder if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir "logs"
    echo "Created 'logs' folder."
fi

# iterate through our settings, indexed by this integer
for ((i = 0; i < 32; i++)); do

    # launch our job
    sbatch analyzer_main_runscript.sh $i
    sleep 0.5

done

