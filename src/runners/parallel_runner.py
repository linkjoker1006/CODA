from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import imageio
import numpy as np
import time
import copy
import os


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i
            env_args[i]["common_reward"] = self.args.common_reward
            env_args[i]["reward_scalarisation"] = self.args.reward_scalarisation
        self.ps = [
            Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))),
            )
            for env_arg, worker_conn in zip(env_args, self.worker_conns)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.last_test_stats = {}
        self.train_infos = []
        self.test_infos = []
        
        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        if self.args.use_cuda and not self.args.cpu_inference:
            self.batch_device = self.args.device
        else:
            self.batch_device = "cpu" if self.args.buffer_cpu_only else self.args.device
        print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& self.batch_device={}".format(
            self.batch_device))
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.batch_device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        if (self.args.use_cuda and self.args.cpu_inference) and str(self.mac.get_device()) != "cpu":
            self.mac.cpu()  # copy model to cpu

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        
        self.batch.update(pre_transition_data, ts=0, mark_filled=True)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        # TODO: 画图功能
        if test_mode and self.args.save_replay:
            self.frames_list = [[] for _ in range(self.batch_size)]
        
        self.reset()

        all_terminated = False
        if self.args.common_reward:
            episode_returns = [0 for _ in range(self.batch_size)]
        else:
            episode_returns = [
                np.zeros(self.args.n_agents) for _ in range(self.batch_size)
            ]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            # TODO: 画图功能
            if test_mode and self.args.save_replay:
                for parent_conn in self.parent_conns:
                    parent_conn.send(("save_replay", None))
                for i in range(len(self.parent_conns)):
                    img = parent_conn.recv()
                    self.frames_list[i].append(img)
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if self.args.mac == "n_mac_coda":
                actions, ask_actions, advice_actions, chosen_actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                cpu_ask_actions = ask_actions.to("cpu").numpy()  # (batch_size, n_agents)
                cpu_advice_actions = advice_actions.to("cpu").numpy()  # (batch_size, n_agents)
                cpu_chosen_actions = chosen_actions.to("cpu").numpy()  # (batch_size, n_agents)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                ask_actions = None
                advice_actions = None
                chosen_actions = None
            cpu_actions = actions.to("cpu").numpy()  # (batch_size, n_agents)
            # Update the actions taken
            actions_chosen = {"actions": np.expand_dims(cpu_actions, axis=1)}
            if ask_actions is not None:
                actions_chosen["ask_actions"] = np.expand_dims(cpu_ask_actions, axis=1)
                actions_chosen["advice_actions"] = np.expand_dims(cpu_advice_actions, axis=1)
                actions_chosen["chosen_actions"] = np.expand_dims(cpu_chosen_actions, axis=1)
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }
            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    if self.args.common_reward and ("pgm" in self.args.env or "ft" in self.args.env or "cleanup" in self.args.env):
                        data["reward"] = sum(data["reward"])
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        cur_infos = self.test_infos if test_mode else self.test_infos
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        # 1. 获取所有信息字典中的全部唯一键
        all_keys = set().union(*(d.keys() for d in infos))
        # 2. 遍历每一个键
        for k in all_keys:
            # 3. 从infos列表中找到这个键对应的第一个非空值，以判断其类型
            #    使用生成器表达式和 next() 可以高效地实现这一点
            first_value = next((d.get(k) for d in infos if d.get(k) is not None), None)

            # 4. 根据值的类型进行不同的处理
            if isinstance(first_value, list):
                # 如果值是列表，则按位相加
                list_len = len(first_value)
                # 提取所有字典中该键对应的列表，如果某个字典没有该键，则使用一个全零列表作为默认值
                lists_to_sum = [d.get(k, [0] * list_len) for d in infos]
                # 使用 zip(*...) 将列表的各个元素打包在一起，然后求和
                cur_stats[k] = [sum(elements) for elements in zip(*lists_to_sum)]

            elif first_value is not None:
                # 如果是数值或其他可求和的类型，则直接求和
                # (和你的原始逻辑相同)
                cur_stats[k] = sum(d.get(k, 0) for d in infos)

        # 原来的写法
        # cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)
        # print(cur_stats["test_info"])

        cur_returns.extend(episode_returns)
        if "test_info" in final_env_infos[0]:
            tmp = []
            for i in range(len(final_env_infos)):
                tmp.append(final_env_infos[i]["test_info"])
            cur_infos.extend(np.array(tmp))
        
        # TODO: 画图功能
        if test_mode and self.args.save_replay:
            for i in range(self.batch_size):
                # 保存GIF
                imageio.mimsave('{}_{}.gif'.format(self.args.env, cur_stats["n_episodes"]), self.frames_list[i], fps=5)
                # 这是第 i 个 batch/episode 的帧列表
                episode_frames = self.frames_list[i]
                
                # 1. 创建一个唯一的目录名来存放这些帧
                # 我们在原有的基础上加入了 batch 索引 'i'，以防 batch_size > 1 时发生覆盖
                episode_id = cur_stats["n_episodes"]
                save_dir = '{}_{}_{}_b{}'.format(self.args.env, self.args.name, episode_id, i)
                
                # 2. 创建目录 (如果它不存在)
                os.makedirs(save_dir, exist_ok=True)
                
                # 3. 遍历这一集中的每一帧并保存
                for frame_index, frame_image in enumerate(episode_frames):
                    # 4. 创建每一帧的文件名
                    # 'f{:04d}' 会将帧编号格式化为4位数 (例如: 0000, 0001, 0010)
                    # 这样可以确保文件按正确的顺序排序
                    filename = os.path.join(save_dir, 'frame_{:04d}.png'.format(frame_index))
                    
                    # 5. 保存单张图片
                    imageio.imwrite(filename, frame_image)

        # test with chunksize
        # chunksize = 32
        # np.set_printoptions(linewidth=np.inf)
        # if cur_stats['n_episodes'] % chunksize == 0:
        #     print(len(self.test_returns))
        #     if cur_stats['n_episodes'] / chunksize == 1:
        #         if self.args.env in ["sc2", "sc2_v2"]:
        #             print("test_battle_won: {}".format(cur_stats['battle_won'] / chunksize))
        #         else:
        #             print("info_mean: {}".format(np.mean(cur_infos, axis=0)))
        #             if self.args.common_reward:
        #                 print("return_mean: {}".format(np.mean(cur_returns)))
        #             else:
        #                 print("return_mean: {}".format(np.mean(np.sum(cur_returns, axis=-1))))
        #         self.last_test_stats = copy.deepcopy(cur_stats)
        #     else:
        #         if self.args.env in ["sc2", "sc2_v2"]:
        #             print("test_battle_won: {}".format((cur_stats['battle_won'] - self.last_test_stats['battle_won']) / chunksize))
        #         else:
        #             print("info_mean: {}".format(np.mean(cur_infos[-chunksize:], axis=0)))
        #             if self.args.common_reward:
        #                 print("return_mean: {}".format(np.mean(cur_returns[-chunksize:])))
        #             else:
        #                 print("return_mean: {}".format(np.mean(np.sum(cur_returns[-chunksize:], axis=-1))))
        #         self.last_test_stats = copy.deepcopy(cur_stats)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("eval/epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def save_replay(self):
        print("----------------------------Replay----------------------------")
        if self.args.save_replay:
            for parent_conn in self.parent_conns:
                parent_conn.send(("save_replay", None))
            for parent_conn in self.parent_conns:
                _ = parent_conn.recv()

    def _log(self, returns, stats, prefix):
        if "test_info" in stats:
            for i in range(len(stats["test_info"])):
                    stats["test_info"][i] = stats["test_info"][i] / len(self.test_returns)
            print("eval/test_total_info_mean: {}".format(stats["test_info"]))
        if self.args.common_reward:
            self.logger.log_stat("eval/" + prefix + "total_return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat("eval/" + prefix + "total_return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    "eval/" + prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    "eval/" + prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                "eval/" + prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                "eval/" + prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes" and k != "test_info":
                self.logger.log_stat("eval/" + prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
        print(self.logger.stats)


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            _, reward, terminated, truncated, env_info = env.step(actions)
            terminated = terminated or truncated
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs" : obs
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "save_replay":
            remote.send(env.save_replay())
        elif cmd == "render":
            env.render()
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
        
