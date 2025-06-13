-- Remove agents from the database.
-- Keep agents with id <= 7, as these are our example agents.
DELETE FROM agentdb WHERE id > 7;
-- Remove any pending prompt injections for agents.
DELETE FROM agentpromptinject WHERE 1 = 1;
-- Remove history of messages from existing NFTGame agents.
DELETE FROM long_term_memories WHERE task_description LIKE 'nft-%';
-- Remove agent's self-modified prompts from existing NFTGame agents.
DELETE FROM prompts WHERE session_identifier LIKE 'nft-%';
-- Delete reports from last games.
DELETE FROM report_nft_game WHERE 1 = 1;
-- Delete game run configs.
DELETE FROM nft_game_round WHERE 1 = 1;