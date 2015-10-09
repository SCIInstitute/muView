#!/bin/sh

if [ ! -n "$1" ]
then
  echo "Usage: `basename $0` argument1 argument2 etc."
  exit $E_BADARGS
fi  

echo "Range [$1, $2]"
echo "Input 1: $3"
echo "Input 2: $4"
echo "Input 3: $5"
echo "Output: $6"

rm output.txt
rm $6

for ((a=$1; a <= $2; a++))
do

  echo Isoval=$a.0
  echo "./dfield.exe -create -min -isoval $a tmp.min_$a.0.df $3 $4 $5 >> output.txt"
  ./dfield.exe -create -min -isoval $a tmp.min_$a.0.df $3 $4 $5 >> output.txt
  echo "./dfield.exe -create -max -isoval $a tmp.min_$a.0.df $3 $4 $5 >> output.txt"
  ./dfield.exe -create -max -isoval $a tmp.max_$a.0.df $3 $4 $5 >> output.txt

  echo Isoval=$a.5
  echo "./dfield.exe -create -min -isoval $a.5 tmp.min_$a.5.df $3 $4 $5 >> output.txt"
  ./dfield.exe -create -min -isoval $a.5 tmp.min_$a.5.df $3 $4 $5 >> output.txt
  echo "./dfield.exe -create -max -isoval $a.5 tmp.max_$a.5.df $3 $4 $5 >> output.txt"
  ./dfield.exe -create -max -isoval $a.5 tmp.max_$a.5.df $3 $4 $5 >> output.txt

done                       

./dfield.exe -merge -output $6 tmp.*.df tmp.*.df >> output.txt
rm tmp.*.df
