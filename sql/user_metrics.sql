SELECT
    user_id,
    exp_group,
    sum(action = 'view') AS views,
    sum(action = 'like') AS likes,
    likes / views AS ctr
FROM {db}.feed_actions
WHERE exp_group IN ({groups})
  AND toDate(time) BETWEEN '{start}' AND '{end}'
GROUP BY user_id, exp_group
