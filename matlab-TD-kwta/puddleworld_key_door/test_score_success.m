%save_episodes = [1,1000,5000,10000,20000,50000,100000,200000,300000,400000];
save_episodes = 100000;
for ep=save_episodes
    filename = ['test_nh_221/weights',int2str(ep),'.mat'];
    load(filename);
    [successful_key_door_episodes, successful_key_episodes, scores_vec, total_episodes] = test_score_success_func(ep,Wih, biasih, Who, biasho);
end
