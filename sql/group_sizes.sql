SELECT
    exp_group,
    uniqExact(user_id) AS users
FROM {db}.feed_actions
WHERE exp_group IN ({groups})
  AND toDate(time) BETWEEN '{start}' AND '{end}'
GROUP BY exp_group
ORDER BY exp_group
