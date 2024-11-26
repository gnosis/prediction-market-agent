if __name__ == "__main__":
    with Cache.disk(cache_path_root="/tmp/autogen_cache_app2") as cache:
        code_executor_agent.initiate_chat(
            code_writer_agent,
            message=user_input,
            cache=cache,
        )
