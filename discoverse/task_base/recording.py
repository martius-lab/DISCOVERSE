import multiprocessing as mp
import os
import subprocess

class Recording:
    def __init__(self, save_dir, cfg, start_idx=0):
        self.save_dir = save_dir
        self.cfg = cfg
        self.data_idx = start_idx
        self.act_lst = []
        self.obs_lst = []
        self.state_lst = []
        self.process_list = []

    def add(self, act, obs, state):
        self.act_lst.append(act)
        self.obs_lst.append(obs)
        self.state_lst.append(state)

    def reset(self):
        self.act_lst = []
        self.obs_lst = []
        self.state_lst = []

    def record_episode(self, record_function, success=True):
        save_path = os.path.join(self.save_dir, "{:03d}".format(self.data_idx))
        # Ensure directory exists - though usually created by the environment or before
        os.makedirs(save_path, exist_ok=True)

        process = mp.Process(
            target=record_function,
            args=(save_path, self.act_lst, self.obs_lst, self.cfg, self.state_lst, True),
        )
        process.start()
        self.process_list.append(process)
        if success:
            self.data_idx += 1

        # We don't necessarily reset here if the loop does it explicitly,
        # but it's good practice to clear references.
        # However, the user's loop calls reset() on the simulation which triggers list clearing.
        # We'll let the user call reset() or rely on this.
        self.reset()

    def finish_recording(self):
        for p in self.process_list:
            p.join()

    def export_to_lerobot(self):
        print("\n" + "=" * 60)
        print("Rendering observations from recorded states...")
        print("=" * 60)

        playback_script = os.path.join(self.save_dir, "record_playback.py")
        if os.path.exists(playback_script):
            # Run playback script for all trajectories
            result = subprocess.run(["python", playback_script, "--all"], cwd=self.save_dir)
            if result.returncode == 0:
                print("\n" + "=" * 60)
                print("Observation rendering complete!")
                print("=" * 60)
            else:
                print(
                    f"\nWarning: Playback script exited with code {result.returncode}"
                )
        else:
            print(f"Warning: Playback script not found at {playback_script}")
