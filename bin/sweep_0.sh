#for i in 10800 10801 10802 10803 10804
#do
    #CUDA_VISIBLE_DEVICES=0 \
        #python dlib_reproduce_pyversion.py --output_directory /project/xqzhu_dis/repo_results/dis_lib_reproduce/${i} --model_num ${i} &
#done
#wait

#for i in 10900 10902 10904 10906 10908
for i in 10900
do
    for j in 0 1 2 3 4
    do
        k=$(( $i + $j ))
        CUDA_VISIBLE_DEVICES=0 \
            python dlib_reproduce_pyversion.py --output_directory /project/xqzhu_dis/repo_results/dis_lib_reproduce/${k} --model_num ${k} &
        #echo ${k}
    done
    wait
done

