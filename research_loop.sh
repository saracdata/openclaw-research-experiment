
#!/bin/bash



PROJECT_DIR="/root/quant"

cd "$PROJECT_DIR" || exit 1



MAX_HOURS=24

END_TIME=$(($(date +%s) + MAX_HOURS * 3600))

ITERATION=1



echo "=========================================="

echo " Starting 24-Hour Quantitative Research Loop"

echo " Started at: $(date)"

echo " Ending at:  $(date -d "@$END_TIME")"

echo "=========================================="



while [ $(date +%s) -lt $END_TIME ]; do

    echo "------------------------------------------"

    echo "Starting Research Iteration #$ITERATION [$(date)]"

    echo "------------------------------------------"



    PROMPT="Perform iteration #$ITERATION of quantitative research on https://www.quantstart.com/articles/Beginners-Guide-to-Quantitative-Trading/. Read the concepts, design new backtesting experiments or strategy parameters in python, execute the scripts, write all generated csv/png/md outputs to the quant folder, and update REPORT.md with new findings."



    openclaw agent --local --agent main --message "$PROMPT"



    git add .

    git commit -m "Auto-research iteration #$ITERATION [$(date +'%Y-%m-%d %H:%M')]"

    git push origin main || echo "Git push failed, retrying on next iteration..."



    echo "Completed Iteration #$ITERATION."

    ITERATION=$((ITERATION + 1))



    echo "Sleeping for 120 seconds before next iteration..."

    sleep 120

done



echo "24-Hour Research Loop finished at $(date)"

