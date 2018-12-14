import paths


ENV_1A_EASY = "1a_easy"
ENV_1A_3_BLOCKS = "1a_3_blocks"
ENV_1A_MEDIUM = "1a_medium"
ENV_4A_EASY = "4a_easy"
ENV_4A_MEDIUM = "4a_medium"

ENVS = [ENV_1A_EASY, ENV_1A_3_BLOCKS, ENV_1A_MEDIUM, ENV_4A_EASY, ENV_4A_MEDIUM]


def env_to_path(env):

    assert env in ENVS

    if env == ENV_1A_EASY:
        return paths.ENV_1A_EASY
    elif env == ENV_1A_3_BLOCKS:
        return paths.ENV_1A_3_BLOCKS
    elif env == ENV_1A_MEDIUM:
        return paths.ENV_1A_MEDIUM
    elif env == ENV_4A_EASY:
        return paths.ENV_4A_EASY
    else:
        return paths.ENV_4A_MEDIUM
