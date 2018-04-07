
save_episodes = 5000;
ep = save_episodes;
%for ep=save_episodes
    %filename = ['test_nh_884/weights',int2str(ep),'.mat'];
    filename = 'weights15548.mat';
%    load(filename);
    [successful_key_door_episodes, successful_key_episodes, scores_vec, total_episodes] = test_score_success_func(ep,Wih, biasih, Who, biasho);
%end

length(successful_key_door_episodes)
length(successful_key_episodes)
mean(scores_vec)
