def perform_benchmarking(self, agents_wrapper: AgentsWrapper, number_of_runs=10):
        methods = ["Bottlenecking", "Direct_Tanh", "Direct_OneHot"]
        benchmark_results = []
        sizes_bottlenecking = [1, 2]
        sizes_direct_tanh = [1, 2]
        sizes_direct_onehot = [3, 5, 9]
        agents_wrapper.train_agent(2001) # We train only a single network
        original_rl_reward = agents_wrapper.agent.evaluation_result.returns[-1]
        original_rl_reachability = agents_wrapper.agent.evaluation_result.reach_probs[-1]
        agents_wrapper.agent.set_agent_greedy()
        method_sizes_map = {
            "Bottlenecking": sizes_bottlenecking,
            "Direct_Tanh": sizes_direct_tanh,
            "Direct_OneHot": sizes_direct_onehot
        }
        # agents_wrapper.agent.load_agent(True)
        
        for i in range(number_of_runs):
            for method, sizes in method_sizes_map.items():
                for size in sizes:
                    if method == "Bottlenecking":
                        self.use_one_hot_memory = False
                        bottleneck_extractor = BottleneckExtractor(
                            agents_wrapper.agent.tf_environment, input_dim=64, latent_dim=size)
                        bottleneck_extractor.train_autoencoder(
                            agents_wrapper.agent.wrapper, num_epochs=80, num_data_steps=(self.args.max_steps + 1) * 6)
                        extracted_fsc = bottleneck_extractor.extract_fsc(
                            policy=agents_wrapper.agent.wrapper, environment=agents_wrapper.agent.environment)
                        assignment = self.compute_paynt_assignment_from_fsc_like(
                            extracted_fsc, latent_dim=size, agents_wrapper=agents_wrapper)
                        logger.info(
                            f"Benchmarking {method} with size {size} finished. Verified performance: {self.quotient.specification.optimality.optimum}.")
                        # paynt_export = self.quotient.extract_policy(assignment)
                        paynt_bounds = self.quotient.specification.optimality.optimum
                        benchmark_result = ExtractionBenchmarkRes(
                            type=method, memory_size=size, accuracies=[0.0], verified_performance=paynt_bounds,
                            original_rl_reward=original_rl_reward, original_rl_reachability=original_rl_reachability,
                            reachabilities=[0.0], rewards=[0.0])
                    else:  # Direct_Tanh or Direct_OneHot
                        self.use_one_hot_memory = True if method == "Direct_OneHot" else False
                        extracted_fsc, stats = self.perform_rl_to_fsc_cloning(
                            agents_wrapper.agent.wrapper, agents_wrapper.agent.environment, agents_wrapper.agent.tf_environment, latent_dim=size)

                        assignment = self.compute_paynt_assignment_from_fsc_like(
                            extracted_fsc, latent_dim=size, agents_wrapper=agents_wrapper)
                        logger.info(f"Exporting assignment.")
                        # paynt_export = self.quotient.extract_policy(assignment)
                        paynt_bounds = self.quotient.specification.optimality.optimum

                        benchmark_result = ExtractionBenchmarkRes(
                            type=method, memory_size=size, accuracies=stats.evaluation_accuracies, verified_performance=paynt_bounds,
                            original_rl_reward=original_rl_reward, original_rl_reachability=original_rl_reachability,
                            reachabilities=stats.extracted_policy_reachabilities, rewards=stats.extracted_policy_rewards)
                    benchmark_results.append(benchmark_result)
                    self.quotient.specification.reset()
                    logger.info(
                        f"Benchmarking {method} with size {size} finished. Verified performance: {paynt_bounds}.")
                    ExtractionBenchmarkResManager.create_folder_with_extraction_benchmark_res(
                        f"experiments_extraction/{self.model_name}", benchmark_results)
        return benchmark_results