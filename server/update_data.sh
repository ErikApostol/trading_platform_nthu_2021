#!/bin/bash

TICKERS=$( python3 get_unique_tickers.py | sed "s/', '/ /g" | sed "s/{'/( /g" | sed "s/'}/ )/g" ) # Bash array
NUM_ASSETS="${#TICKERS[@]}" # Array length
OLDFILE=data_for_trading_platform.csv
NEWFILE=$OLDFILE.new
HEADER='Date,Ticker,Open,High,Low,Close,Adj Close,Volume'

echo $HEADER > $NEWFILE

i=1
for TICKER in "${TICKERS[@]}"
do
    echo Downloading $TICKER $i/$NUM_ASSETS
    curl 'https://query1.finance.yahoo.com/v7/finance/download/'$TICKER'?period1=1420070400&period2=2000000000&interval=1d&events=history&includeAdjustedClose=true' | tail --lines=+2 | sed 's/^[0-9\-]\+,/\0'$TICKER',/g' >> $NEWFILE
    echo ""
    i=$(($i+1))
done

# Remove lines containing "null"
sed -i '/null/d' $NEWFILE
# Remove empty lines
sed -i '/^$/d' $NEWFILE

# Human review
vim $NEWFILE 
mv -i $NEWFILE $OLDFILE
