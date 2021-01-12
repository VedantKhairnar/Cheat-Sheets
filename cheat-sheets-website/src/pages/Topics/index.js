import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './style.scss';
import Topic from '../Topic';

function Topics() {

    const [topics, setTopics] = useState([]);
    const [isTopic, setIsTopic] = useState(false);
    const [status, setStatus] = useState("normal");

    useEffect(() => {
        setStatus("loading");
        axios.get('https://api.github.com/repos/VedantKhairnar/Cheat-Sheets/contents')
            .then(response => {
                setStatus("success");
                setTopics(response.data);
            })
            .catch(err => {
                console.log(err);
                setStatus("error");
            });
    }, []);

    const [topic, setTopic] = useState({});

    function topicClickHandle(topic) {
        setTopic(topic);
        setIsTopic(current => !current);
    }

    if (status === 'loading') {
        return (
            <div className="Topics">Loading...</div>
        )
    }

    if (status === 'error') {
        return (
            <div className="Topics">Error!</div>
        )
    }

    return (
        <>
            {
                !isTopic ?
                    <div className="Topics">
                        <div className="Topics-title">
                            Topics
            </div>
                        <div className="Topics-grid">
                            {
                                topics.map(topic => (
                                    topic.type === 'dir' && topic.name !== 'cheat-sheets-website' && <div className="Topics-Button" onClick={() => topicClickHandle(topic)} key={topic.name}>{topic.name}</div>
                                ))
                            }
                        </div>
                    </div> :
                    <Topic topic={topic} setIsTopic={setIsTopic} />
            }
        </>
    )
}

export default Topics
