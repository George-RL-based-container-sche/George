Num_C_per_app_total = 100;
filename = "./batch_set_cpo_27node_"+int2str(Num_C_per_app_total)+".csv";
preference = table2array(readtable(filename));
allocation_set =[];
ratio_set = [];
abs_v_set = [];
no_app = 7;
time_cost = [];
weight_cpo = 1;

for choice = 1:30
    fprintf("newchoice: ")
    choice
    fprintf("weight_cpo: ")
    weight_cpo
    alloc = [];
    rat = [];
    abs_v = [];
    tic();
    Num_C_per_app_1 = floor(preference(choice,:));
    [allocation,ratio,abs_vio] = medea_core(Num_C_per_app_1, 27,weight_cpo);
    alloc = [alloc; allocation];
    rat = [rat; ratio];
    abs_v = [abs_v;abs_vio];
    time_2 = toc();
  
    alloc_reslut = reshape(alloc,[1,27*7]);
    allocation_set = [allocation_set; alloc_reslut];
    abs_v_set = [abs_v_set; mean(abs_v)];
    ratio_set = [ratio_set; mean(rat)];
    time_cost = [time_cost;time_2]
end
    csvwrite("./results/batch_set_cpo_"+int2str(Num_C_per_app_total)+"_"+int2str(weight_cpo)+"_allocation.csv",allocation_set);
    csvwrite("./results/batch_set_cpo_"+int2str(Num_C_per_app_total)+"_"+int2str(weight_cpo)+"_ratio.csv",ratio_set);
    csvwrite("./results/batch_set_cpo_"+int2str(Num_C_per_app_total)+"_"+int2str(weight_cpo)+"_abs_vio.csv",abs_v_set);
