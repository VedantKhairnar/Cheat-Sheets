import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { BiArrowBack } from 'react-icons/bi';
import './style.scss';

function Topic(props) {

    const [data, setData] = useState([]);
    const [status, setStatus] = useState("normal");

    useEffect(() => {
        setStatus("loading");
        axios.get(props.topic.url)
            .then(response => {
                setData(response.data);
                setStatus("normal");
            })
            .catch(err => {
                console.log(err);
                setStatus("error");
            });
    }, [props.topic.url])

    function goBack() {
        props.setIsTopic(current => !current);
    }

    return (
        <div className="Topic">
            <div className="Topic-back" onClick={goBack}><BiArrowBack /></div>
            {
                status === "normal" ?
                    <>
                        <div className="Topic-title">{props.topic.name}</div>
                        <div className="Topic-body">
                            {
                                data.map(d => (
                                    <a key={d.name} className="Topic-Link" href={d.html_url} target="_blank" rel="noreferrer">{d.name}</a>
                                ))
                            }
                            {
                                data.length === 0 &&
                                <div className="Topic-Link">No Cheat Sheets Found.</div>
                            }
                        </div>
                    </> :

                    status === "loading" ?
                        <div>Loading...</div> :
                        <div>Error! Go back, or refresh the page.</div>
            }
        </div>
    )
}

export default Topic
