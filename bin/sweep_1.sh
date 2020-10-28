#for i in 10805 10806 10807 10808 10809
#do
    #CUDA_VISIBLE_DEVICES=1 \
        #python dlib_reproduce_pyversion.py --output_directory /project/xqzhu_dis/repo_results/dis_lib_reproduce/${i} --model_num ${i} &
#done
#wait

for i in 10850
do
    for j in 0 1 2 3 4 5 6 7 8 9
    do
        k=$(( $i + $j ))
        CUDA_VISIBLE_DEVICES=1 \
            python dlib_reproduce_pyversion.py --output_directory /project/xqzhu_dis/repo_results/dis_lib_reproduce/${k} --model_num ${k}
        #echo ${k}
    done
    wait
done

